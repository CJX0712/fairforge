"""FairForge command line interface.

Usage:
  fairforge run    [--scenario S] [--mitigator M] [--seed N] [--out PATH]
  fairforge data   [--scenario S] [--n-per-group N] [--seed N]
  fairforge backend

Author: 晨星
"""

import argparse
import json
import logging
import sys


def _cmd_run(args: argparse.Namespace) -> int:
    from fairforge.data.generator import SCENARIOS, generate_scenario
    from fairforge.metrics.fairness import fairness_report, sanitize_metrics
    from fairforge.metrics.performance import accuracy as accuracy_metric
    from fairforge.pipeline.benchmark import run_single, save_benchmark

    scenarios = [args.scenario] if args.scenario else list(SCENARIOS)
    mitigators = [args.mitigator] if args.mitigator else None
    entries = []
    for scenario in scenarios:
        dataset = generate_scenario(scenario, seed=args.seed)
        for m_name in mitigators or ["reweighing", "exponentiated_gradient", "threshold_optimizer"]:
            m_obj, y_pred, elapsed = run_single(dataset, m_name, seed=args.seed)
            entries.append(
                {
                    "scenario": scenario,
                    "mitigator": m_name,
                    "backend": m_obj.resolved_backend,
                    "used_fallback": m_obj.used_fallback,
                    "accuracy": float(accuracy_metric(dataset.y, y_pred)),
                    "metrics": sanitize_metrics(fairness_report(dataset.y, y_pred, dataset.A)),
                    "elapsed_s": round(elapsed, 4),
                    "seed": args.seed,
                }
            )
    save_benchmark(entries, args.out)
    print(json.dumps({"n_entries": len(entries), "out": str(args.out)}, indent=2))
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    from fairforge.data.generator import generate_scenario

    ds = generate_scenario(args.scenario, n_per_group=args.n_per_group, seed=args.seed)
    info = {
        "scenario": ds.scenario,
        "n_samples": ds.n_samples,
        "n_features": ds.n_features,
        "group_counts": {str(g): int((ds.A == g).sum()) for g in (0, 1)},
        "positive_rate": float(ds.y.mean()),
        "counterfactual_positive_rate": float(ds.y_cf.mean()) if ds.y_cf is not None else None,
    }
    print(json.dumps(info, indent=2))
    return 0


def _cmd_backend(args: argparse.Namespace) -> int:
    from fairforge.core.backend import available_backends, backend_version

    print(
        json.dumps(
            {
                "available": available_backends(refresh=True),
                "versions": {
                    "fairlearn": backend_version("fairlearn"),
                    "optuna": backend_version("optuna"),
                },
                "environment": {"python": sys.version.split()[0]},
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fairforge", description="FairForge fairness toolkit (晨星)"
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run mitigators on scenarios, write benchmark JSON")
    p_run.add_argument(
        "--scenario",
        choices=sorted(__import__("fairforge.data.generator", fromlist=["SCENARIOS"]).SCENARIOS),
    )
    p_run.add_argument("--mitigator", default=None)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--out", default="benchmark.json")
    p_run.set_defaults(func=_cmd_run)

    p_data = sub.add_parser("data", help="generate and inspect a benchmark scenario")
    p_data.add_argument("--scenario", default="mixed")
    p_data.add_argument("--n-per-group", type=int, default=600)
    p_data.add_argument("--seed", type=int, default=42)
    p_data.set_defaults(func=_cmd_data)

    p_be = sub.add_parser("backend", help="show detected optional backends")
    p_be.set_defaults(func=_cmd_backend)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
