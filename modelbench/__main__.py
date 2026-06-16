"""CLI: python -m modelbench run experiments/example.yaml
        python -m modelbench calibrate experiments/labelled.yaml --judge openai/gpt-5
"""

from __future__ import annotations

import argparse
import asyncio


def _cmd_run(args) -> None:
    from .config import load_experiment, run_experiment

    exp = load_experiment(args.experiment)
    out = asyncio.run(run_experiment(exp))

    errors = [r for r in out["results"] if not r.ok]
    print(f"\nRan {len(out['results'])} cells ({len(errors)} errored).")
    if errors:
        for r in errors[:10]:
            print(f"  ERROR {r.key}: {r.error}")

    total_cost = sum(r.cost_usd or 0 for r in out["results"])
    print(f"Total cost: ${total_cost:.4f}\n")

    print(f"Leaderboard (axis={exp.axis}, swap-tested pairwise):")
    for s in out["standings"]:
        print(
            f"  {s.label:40s}  win-rate {s.win_rate:5.1%}  "
            f"(W{s.wins} L{s.losses} T{s.ties})"
        )


def _cmd_models(args) -> None:
    from .client import OpenRouterClient

    models = asyncio.run(OpenRouterClient().list_models())
    needle = (args.filter or "").lower()
    rows = [m for m in models if needle in m.get("id", "").lower()]
    rows.sort(key=lambda m: m.get("id", ""))
    print(f"\n{len(rows)} models matching {args.filter!r}:\n")
    for m in rows:
        pricing = m.get("pricing", {}) or {}
        # OpenRouter prices are $ per token; show per 1M for readability.
        try:
            pin = float(pricing.get("prompt", 0)) * 1_000_000
            pout = float(pricing.get("completion", 0)) * 1_000_000
            price = f"${pin:.2f}/${pout:.2f} per 1M in/out"
        except (TypeError, ValueError):
            price = "price n/a"
        print(f"  {m.get('id', '?'):45s}  {price}")


def _cmd_calibrate(args) -> None:
    from .calibrate import calibrate

    report = asyncio.run(calibrate(args.labelled_set, args.judge))
    print(f"\nJudge: {args.judge}   pairs: {report['n']}")
    print(f"  agreement with your labels : {report['agreement']:.1%}")
    print(f"  position consistency       : {report['position_consistency']:.1%}")
    if report["agreement"] < 0.8:
        print("\n  ⚠ agreement < 80% — don't trust this judge yet. Fix rubric or swap model.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="modelbench")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an experiment YAML")
    run.add_argument("experiment")
    run.set_defaults(func=_cmd_run)

    mod = sub.add_parser("models", help="list available OpenRouter models (+ pricing)")
    mod.add_argument("filter", nargs="?", default="", help="substring to filter by, e.g. claude")
    mod.set_defaults(func=_cmd_models)

    cal = sub.add_parser("calibrate", help="validate a judge against labelled pairs")
    cal.add_argument("labelled_set")
    cal.add_argument("--judge", required=True, help="judge model slug")
    cal.set_defaults(func=_cmd_calibrate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
