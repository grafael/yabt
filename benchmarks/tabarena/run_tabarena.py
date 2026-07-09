"""Benchmark YABT on TabArena-Lite and compare against the leaderboard.

Usage (from the tabarena venv):
    python run_tabarena.py            # 3-dataset smoke run (quickstart datasets)
    python run_tabarena.py --full     # all TabArena-Lite tasks (one split each)

Results cache under ./experiments/, leaderboard + figures under ./eval/.
Re-runs reuse cached fold results, so a crashed run resumes where it left off.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from yabt_model import YABTModel  # noqa: E402

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle  # noqa: E402
from tabarena.contexts import TabArenaContext  # noqa: E402

SMOKE_DATASETS = ["blood-transfusion-service-center", "QSAR_fish_toxicity", "anneal"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run all TabArena-Lite tasks")
    parser.add_argument("--n-configs", type=int, default=0, help="extra random HPO configs")
    args = parser.parse_args()

    run_name = "yabt_tabarena_full" if args.full else "yabt_tabarena_smoke"
    results_dir = str(HERE / "experiments" / run_name)
    eval_dir = HERE / "eval" / run_name

    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(YABTModel.config_generator(), args.n_configs)],
    ).build_experiments()

    context = TabArenaContext()
    build_kwargs = {} if args.full else {"dataset_names": SMOKE_DATASETS}
    context.build_and_run_jobs(
        experiments,
        expname=results_dir,
        subset="lite",
        build_kwargs=build_kwargs,
        new_result_prefix="[New] ",
        debug_mode=True,  # in-process backend (no Ray)
    )

    leaderboard = context.compare(output_dir=eval_dir)
    leaderboard_website = context.leaderboard_to_website_format(leaderboard=leaderboard)
    print("\n=== TabArena leaderboard (website format) ===")
    print(leaderboard_website.to_markdown(index=False))
    print(f"\nFigures/CSVs in {eval_dir}")


if __name__ == "__main__":
    main()
