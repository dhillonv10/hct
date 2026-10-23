"""Command line entry point: ``epimod prepare | train | evaluate``."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path


def _kv(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="epimod", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="download ComputAge Bench, select a CpG panel, cache arrays")
    p.add_argument("--n-total", type=int, default=16000)
    p.add_argument("--n-age", type=int, default=12000)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--keep-benchmark-raw", action="store_true")

    t = sub.add_parser("train", help="train one EpiMod run (study-held-out validation)")
    t.add_argument("--out", required=True)
    t.add_argument("--model", nargs="*", metavar="K=V", help="ModelConfig overrides, e.g. n_layers=0 use_genomic_context=False")
    t.add_argument("--train", nargs="*", metavar="K=V", help="TrainConfig overrides, e.g. epochs=30 aug=False")

    e = sub.add_parser("evaluate", help="score runs + baselines on the held-out ComputAge benchmark studies")
    e.add_argument("--run", nargs="+", required=True, metavar="NAME=DIR")
    e.add_argument("--out", required=True)
    e.add_argument("--no-elasticnet", action="store_true")
    e.add_argument("--no-published", action="store_true")

    a = sub.add_parser("analyze", help="modules, per-condition module shifts and the cross-fitted acceleration head")
    a.add_argument("--run", required=True)
    a.add_argument("--out", required=True)

    x = sub.add_parser("transfer", help="unseen-CpG transfer test for a run trained with model cpg_split=A, use_id_embedding=False")
    x.add_argument("--run", required=True)
    x.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "transfer":
        from .analysis import run_cpg_transfer

        run_cpg_transfer(args.run, args.out)
    elif args.cmd == "analyze":
        from .analysis import run_analysis

        run_analysis(args.run, args.out)
    elif args.cmd == "prepare":
        from .data.pipeline import prepare

        prepare(n_total=args.n_total, n_age=args.n_age, workers=args.workers, keep_benchmark_raw=args.keep_benchmark_raw)
    elif args.cmd == "train":
        from .runner import run_training

        run_training(args.out, _kv(args.model), _kv(args.train))
    elif args.cmd == "evaluate":
        import pandas as pd

        from .runner import run_evaluation

        runs = dict(r.split("=", 1) for r in args.run)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(run_evaluation(runs, Path(args.out), elasticnet=not args.no_elasticnet, published=not args.no_published).round(3))


if __name__ == "__main__":
    main()
