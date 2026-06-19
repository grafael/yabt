/* OpenMP-parallel leaf-wise gradient-boosting tree grower.
 *
 * A C reimplementation of yabt.grow_numba._grow with bit-identical split math
 * (Newton gain, sibling-subtraction histograms, global best-first leaf
 * selection, interaction-aware selection steering, in-place row partition).
 *
 * The point of the C port is multi-core scaling: the Numba grower is a single
 * thread of tight scalar loops, but XGBoost/LightGBM saturate every core. The
 * two hot loops here -- the dense histogram build and the per-feature split
 * search -- are parallelized across features with OpenMP. Each feature owns a
 * disjoint output slice (histogram) / an independent reduction (split search),
 * so the parallelism is lock-free and the per-feature result order is preserved
 * exactly, which keeps the trees bit-identical to the serial/Numba grower.
 *
 * Compiled at import time by yabt/grow_c.py into a cached .so and called via
 * ctypes; see that module for the argument marshalling.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define NEG_INF (-INFINITY)
#define LEAF (-1)

/* Dense feature-parallel histogram for rows[start:end] into out (3,F,B).
 * Each feature f writes only out[:, f, :], so threads never collide.
 *
 * ``binned`` is feature-major (F, n): feature f's codes are the contiguous block
 * binned[f*n : f*n+n]. This keeps each thread's working set to one n-byte column
 * (cache/TLB-local) instead of striding by F across the whole matrix -- ~2.4x
 * faster than row-major here, and the layout every serious GBDT lib uses. */
static void build_hist(const uint8_t *binned, const float *grad,
                       const float *hess, const int64_t *rows, int64_t start,
                       int64_t end, float *out, int F, int B, int64_t n) {
    memset(out, 0, (size_t)3 * F * B * sizeof(float));
#pragma omp parallel for schedule(static)
    for (int f = 0; f < F; f++) {
        float *og = out + (size_t)0 * F * B + (size_t)f * B;
        float *oh = out + (size_t)1 * F * B + (size_t)f * B;
        float *oc = out + (size_t)2 * F * B + (size_t)f * B;
        const uint8_t *col = binned + (size_t)f * n;
        for (int64_t i = start; i < end; i++) {
            int64_t r = rows[i];
            int b = col[r];
            og[b] += grad[r];
            oh[b] += hess[r];
            oc[b] += 1.0f;
        }
    }
}

/* Sparse (CSR-of-non-default-bins) histogram. Kept serial: a row touches an
 * arbitrary set of features, so it is not naturally feature-parallel. The wide
 * sparse path already turns O(n*F) into O(nnz+F); the split search below still
 * parallelizes for those datasets. */
static void build_hist_sparse(const int64_t *indptr, const int32_t *indices,
                              const int32_t *data, const int32_t *default_bin,
                              const float *grad, const float *hess,
                              const int64_t *rows, int64_t start, int64_t end,
                              float *out, double *expl_g, double *expl_h,
                              double *expl_c, int F, int B) {
    memset(out, 0, (size_t)3 * F * B * sizeof(float));
    for (int f = 0; f < F; f++) {
        expl_g[f] = 0.0;
        expl_h[f] = 0.0;
        expl_c[f] = 0.0;
    }
    double G = 0.0, H = 0.0, C = 0.0;
    float *o0 = out + (size_t)0 * F * B;
    float *o1 = out + (size_t)1 * F * B;
    float *o2 = out + (size_t)2 * F * B;
    for (int64_t i = start; i < end; i++) {
        int64_t r = rows[i];
        float g = grad[r], h = hess[r];
        G += g;
        H += h;
        C += 1.0;
        for (int64_t idx = indptr[r]; idx < indptr[r + 1]; idx++) {
            int f = indices[idx];
            int b = data[idx];
            o0[(size_t)f * B + b] += g;
            o1[(size_t)f * B + b] += h;
            o2[(size_t)f * B + b] += 1.0f;
            expl_g[f] += g;
            expl_h[f] += h;
            expl_c[f] += 1.0;
        }
    }
    for (int f = 0; f < F; f++) {
        int df = default_bin[f];
        o0[(size_t)f * B + df] += (float)(G - expl_g[f]);
        o1[(size_t)f * B + df] += (float)(H - expl_h[f]);
        o2[(size_t)f * B + df] += (float)(C - expl_c[f]);
    }
}

/* Per-feature interaction boost: boost[j] = 1 + ib * max over path features p of
 * imat[p,j] (clamped at 0). Cheap (O(F*path_len)); left serial. */
static void compute_boost(const uint8_t *path_row, const float *imat, float ib,
                          float *boost, int F) {
    for (int j = 0; j < F; j++) boost[j] = 0.0f;
    for (int p = 0; p < F; p++) {
        if (path_row[p]) {
            const float *row = imat + (size_t)p * F;
            for (int j = 0; j < F; j++)
                if (row[j] > boost[j]) boost[j] = row[j];
        }
    }
    for (int j = 0; j < F; j++) boost[j] = 1.0f + ib * boost[j];
}

/* Best (true_gain, f, b) for one node; true_gain <= 0 means don't split.
 * Selection ranks gain*boost[f] (interaction steering) but returns the true
 * unboosted gain at the chosen position. Each feature's best is computed in
 * parallel into scratch arrays, then reduced in feature-major order so the
 * tie-break (lowest f, then lowest b) is identical to the serial scan. */
static double best_split(const float *hist, const int32_t *nbins, double lam,
                         double gamma, double mcw, int msl,
                         const uint8_t *fmask, const float *boost, int F, int B,
                         double *fb_sel, double *fb_true, int *fb_b,
                         int *out_f, int *out_b) {
#pragma omp parallel for schedule(dynamic, 64)
    for (int f = 0; f < F; f++) {
        fb_sel[f] = NEG_INF;
        fb_true[f] = NEG_INF;
        fb_b[f] = -1;
        if (!fmask[f]) continue;
        int nb = nbins[f];
        double bo = boost[f];
        const float *hg = hist + (size_t)0 * F * B + (size_t)f * B;
        const float *hh = hist + (size_t)1 * F * B + (size_t)f * B;
        const float *hc = hist + (size_t)2 * F * B + (size_t)f * B;
        double Gt = 0.0, Ht = 0.0, Ct = 0.0;
        for (int b = 0; b < nb; b++) {
            Gt += hg[b];
            Ht += hh[b];
            Ct += hc[b];
        }
        double parent = Gt * Gt / (Ht + lam);
        double GL = 0.0, HL = 0.0, CL = 0.0;
        double loc_sel = NEG_INF, loc_true = NEG_INF;
        int loc_b = -1;
        for (int b = 0; b < nb - 1; b++) {
            GL += hg[b];
            HL += hh[b];
            CL += hc[b];
            double GR = Gt - GL, HR = Ht - HL, CR = Ct - CL;
            if (CL >= msl && CR >= msl && HL >= mcw && HR >= mcw) {
                double gain = 0.5 * (GL * GL / (HL + lam) +
                                     GR * GR / (HR + lam) - parent) - gamma;
                double sel = gain * bo;
                if (sel > loc_sel) {
                    loc_sel = sel;
                    loc_true = gain;
                    loc_b = b;
                }
            }
        }
        fb_sel[f] = loc_sel;
        fb_true[f] = loc_true;
        fb_b[f] = loc_b;
    }
    /* Feature-major reduction: strict '>' keeps the first (lowest-f) max, as the
     * serial Numba scan does. */
    double best_sel = NEG_INF, best_true = NEG_INF;
    int bf = -1, bb = -1;
    for (int f = 0; f < F; f++) {
        if (fb_b[f] >= 0 && fb_sel[f] > best_sel) {
            best_sel = fb_sel[f];
            best_true = fb_true[f];
            bf = f;
            bb = fb_b[f];
        }
    }
    *out_f = bf;
    *out_b = bb;
    if (bf < 0) return NEG_INF;
    return best_true;
}

/* In-place partition rows[start:end]: binned[:,f] <= b to the front.
 * ``binned`` is feature-major (F, n); feature f's column is binned[f*n:]. */
static int64_t partition_rows(const uint8_t *binned, int64_t *rows,
                              int64_t start, int64_t end, int f, int b,
                              int64_t n) {
    const uint8_t *col = binned + (size_t)f * n;
    int64_t i = start, j = end - 1;
    while (i <= j) {
        int64_t r = rows[i];
        if (col[r] <= b) {
            i++;
        } else {
            rows[i] = rows[j];
            rows[j] = r;
            j--;
        }
    }
    return i;
}

/* Hard-routing apply: leaf (node) index per row of raw X (n, F), row-major.
 * Axis splits only ("X[:,f] <= threshold" -> left). Each row walks the tree
 * independently, so rows parallelize with no contention or reduction. Mirrors
 * Tree.apply for trees without kernel splits / soft gates. */
void capply(const float *X, const int64_t *feature, const float *threshold,
            const int64_t *left, const int64_t *right, int64_t n, int F,
            int n_threads, int64_t *out_node) {
#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
#endif
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < n; r++) {
        const float *xr = X + (size_t)r * F;
        int64_t node = 0;
        int64_t f;
        while ((f = feature[node]) != LEAF) {
            node = (xr[f] <= threshold[node]) ? left[node] : right[node];
        }
        out_node[r] = node;
    }
}

/* Returns n_nodes; fills the out_* arrays (size 2*max_leaves+1).
 * ``binned`` is feature-major (F, n) -- see build_hist. */
int cgrow(const uint8_t *binned, const float *grad, const float *hess,
          const uint8_t *fmask, const float *imat, float ib, int use_imat,
          const int64_t *indptr, const int32_t *indices, const int32_t *data,
          const int32_t *default_bin, int use_sparse, const int32_t *nbins,
          int64_t n, int F, int B, float lam, float gamma, float mcw, int msl,
          float lr, int max_leaves, int max_depth, int n_threads,
          int64_t *out_feature, int64_t *out_thr_bin, int64_t *out_left,
          int64_t *out_right, float *out_value, int64_t *out_depth) {
#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
#endif
    int max_nodes = 2 * max_leaves + 1;

    int64_t *feature = out_feature;
    int64_t *thr_bin = out_thr_bin;
    int64_t *left = out_left;
    int64_t *right = out_right;
    float *value = out_value;
    int64_t *node_depth = out_depth;
    for (int i = 0; i < max_nodes; i++) {
        feature[i] = LEAF;
        thr_bin[i] = 0;
        left[i] = -1;
        right[i] = -1;
        value[i] = 0.0f;
        node_depth[i] = 0;
    }

    int64_t *node_start = calloc(max_nodes, sizeof(int64_t));
    int64_t *node_end = calloc(max_nodes, sizeof(int64_t));
    float *hist_store = malloc((size_t)max_nodes * 3 * F * B * sizeof(float));
    uint8_t *path_mask = calloc((size_t)max_nodes * F, sizeof(uint8_t));
    int64_t *rows = malloc((size_t)n * sizeof(int64_t));
    float *boost = malloc((size_t)F * sizeof(float));
    double *expl_g = malloc((size_t)F * sizeof(double));
    double *expl_h = malloc((size_t)F * sizeof(double));
    double *expl_c = malloc((size_t)F * sizeof(double));
    double *fb_sel = malloc((size_t)F * sizeof(double));
    double *fb_true = malloc((size_t)F * sizeof(double));
    int *fb_b = malloc((size_t)F * sizeof(int));

    int64_t *leaf_node = malloc((size_t)max_nodes * sizeof(int64_t));
    float *leaf_gain = malloc((size_t)max_nodes * sizeof(float));
    int64_t *leaf_f = malloc((size_t)max_nodes * sizeof(int64_t));
    int64_t *leaf_b = malloc((size_t)max_nodes * sizeof(int64_t));

    for (int64_t i = 0; i < n; i++) rows[i] = i;
    for (int j = 0; j < F; j++) boost[j] = 1.0f;

    size_t hsz = (size_t)3 * F * B;
    node_start[0] = 0;
    node_end[0] = n;
    node_depth[0] = 0;
    if (use_sparse)
        build_hist_sparse(indptr, indices, data, default_bin, grad, hess, rows,
                          0, n, hist_store, expl_g, expl_h, expl_c, F, B);
    else
        build_hist(binned, grad, hess, rows, 0, n, hist_store, F, B, n);

    /* Node total G,H = sum over bins of feature 0's histogram. */
    double Gsum = 0.0, Hsum = 0.0;
    for (int b = 0; b < B; b++) {
        Gsum += hist_store[(size_t)0 * F * B + 0 * B + b];
        Hsum += hist_store[(size_t)1 * F * B + 0 * B + b];
    }
    value[0] = (float)(-lr * Gsum / (Hsum + lam));
    int n_nodes = 1;

    int f0 = -1, b0 = -1;
    double g0 = best_split(hist_store, nbins, lam, gamma, mcw, msl, fmask, boost,
                           F, B, fb_sel, fb_true, fb_b, &f0, &b0);
    if (node_depth[0] >= max_depth) g0 = NEG_INF;
    leaf_node[0] = 0;
    leaf_gain[0] = (float)g0;
    leaf_f[0] = f0;
    leaf_b[0] = b0;
    int n_active = 1, n_leaves = 1;

    while (n_leaves < max_leaves) {
        int best_k = -1;
        float best_g = 0.0f;
        for (int k = 0; k < n_active; k++) {
            if (leaf_gain[k] > best_g) {
                best_g = leaf_gain[k];
                best_k = k;
            }
        }
        if (best_k < 0) break;

        int64_t nid = leaf_node[best_k];
        int f = (int)leaf_f[best_k];
        int b = (int)leaf_b[best_k];
        int64_t s = node_start[nid], e = node_end[nid];
        int64_t mid = partition_rows(binned, rows, s, e, f, b, n);

        int nl = n_nodes, nr = n_nodes + 1;
        n_nodes += 2;
        int64_t d = node_depth[nid] + 1;
        node_start[nl] = s;
        node_end[nl] = mid;
        node_start[nr] = mid;
        node_end[nr] = e;
        node_depth[nl] = d;
        node_depth[nr] = d;
        memcpy(path_mask + (size_t)nl * F, path_mask + (size_t)nid * F, F);
        memcpy(path_mask + (size_t)nr * F, path_mask + (size_t)nid * F, F);
        path_mask[(size_t)nl * F + f] = 1;
        path_mask[(size_t)nr * F + f] = 1;

        float *h_nid = hist_store + (size_t)nid * hsz;
        float *h_nl = hist_store + (size_t)nl * hsz;
        float *h_nr = hist_store + (size_t)nr * hsz;
        if ((mid - s) <= (e - mid)) {
            if (use_sparse)
                build_hist_sparse(indptr, indices, data, default_bin, grad, hess,
                                  rows, s, mid, h_nl, expl_g, expl_h, expl_c, F, B);
            else
                build_hist(binned, grad, hess, rows, s, mid, h_nl, F, B, n);
            for (size_t i = 0; i < hsz; i++) h_nr[i] = h_nid[i] - h_nl[i];
        } else {
            if (use_sparse)
                build_hist_sparse(indptr, indices, data, default_bin, grad, hess,
                                  rows, mid, e, h_nr, expl_g, expl_h, expl_c, F, B);
            else
                build_hist(binned, grad, hess, rows, mid, e, h_nr, F, B, n);
            for (size_t i = 0; i < hsz; i++) h_nl[i] = h_nid[i] - h_nr[i];
        }

        double gl = 0.0, hl = 0.0, gr = 0.0, hr = 0.0;
        for (int bb = 0; bb < B; bb++) {
            gl += h_nl[(size_t)0 * F * B + 0 * B + bb];
            hl += h_nl[(size_t)1 * F * B + 0 * B + bb];
            gr += h_nr[(size_t)0 * F * B + 0 * B + bb];
            hr += h_nr[(size_t)1 * F * B + 0 * B + bb];
        }
        value[nl] = (float)(-lr * gl / (hl + lam));
        value[nr] = (float)(-lr * gr / (hr + lam));

        feature[nid] = f;
        thr_bin[nid] = b;
        left[nid] = nl;
        right[nid] = nr;
        value[nid] = 0.0f;
        n_leaves += 1;

        if (use_imat) compute_boost(path_mask + (size_t)nl * F, imat, ib, boost, F);
        int glf = -1, glb = -1;
        double glg = best_split(h_nl, nbins, lam, gamma, mcw, msl, fmask, boost,
                                F, B, fb_sel, fb_true, fb_b, &glf, &glb);
        if (d >= max_depth) glg = NEG_INF;
        leaf_node[best_k] = nl;
        leaf_gain[best_k] = (float)glg;
        leaf_f[best_k] = glf;
        leaf_b[best_k] = glb;

        if (use_imat) compute_boost(path_mask + (size_t)nr * F, imat, ib, boost, F);
        int grf = -1, grb = -1;
        double grg = best_split(h_nr, nbins, lam, gamma, mcw, msl, fmask, boost,
                                F, B, fb_sel, fb_true, fb_b, &grf, &grb);
        if (d >= max_depth) grg = NEG_INF;
        leaf_node[n_active] = nr;
        leaf_gain[n_active] = (float)grg;
        leaf_f[n_active] = grf;
        leaf_b[n_active] = grb;
        n_active += 1;
    }

    free(node_start);
    free(node_end);
    free(hist_store);
    free(path_mask);
    free(rows);
    free(boost);
    free(expl_g);
    free(expl_h);
    free(expl_c);
    free(fb_sel);
    free(fb_true);
    free(fb_b);
    free(leaf_node);
    free(leaf_gain);
    free(leaf_f);
    free(leaf_b);
    return n_nodes;
}
