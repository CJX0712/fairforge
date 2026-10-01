"""One-shot demo: benchmark matrix + flagship summary -> benchmark.json.

Author: 晨星
"""

import json
import logging
from pathlib import Path

from fairforge.pipeline.benchmark import run_benchmark, save_benchmark
from fairforge.pipeline.flagship import run_flagship

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
OUT = Path(__file__).resolve().parent.parent / "benchmark.json"


def main() -> None:
    entries = run_benchmark(seed=42, n_per_group=1000)
    flagship = run_flagship(seed=42, n_per_group=1000, n_trials=30, timeout_s=90.0)
    save_benchmark(entries, OUT, extra={"flagship": flagship})
    summary = {
        "benchmark_entries": len(entries),
        "flagship": {
            "n_runs": flagship["n_runs"],
            "mean_hv_improvement_rel": round(flagship["mean_hv_improvement_rel"], 4),
            "mean_coverage": round(flagship["mean_coverage"], 4),
            "fallback_events": flagship["fallback_events"],
            "targets_met": flagship["targets_met"],
        },
        "written_to": str(OUT),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
