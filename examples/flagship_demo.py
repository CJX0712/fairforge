"""Flagship FairPareto demo -> flagship.json.

Author: 晨星
"""

import json
import logging
from pathlib import Path

from fairforge.pipeline.flagship import run_flagship

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
OUT = Path(__file__).resolve().parent.parent / "flagship.json"


def main() -> None:
    summary = run_flagship(seed=42, n_per_group=1000, n_trials=30, timeout_s=90.0)
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_runs": summary["n_runs"],
                "mean_hv_improvement_rel": round(summary["mean_hv_improvement_rel"], 4),
                "mean_coverage": round(summary["mean_coverage"], 4),
                "fallback_rate": round(summary["fallback_rate"], 4),
                "targets_met": summary["targets_met"],
                "written_to": str(OUT),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
