"""Public command-line interface; expensive solves always require an explicit command."""

import argparse
import json
import os


def main(argv=None):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    parser = argparse.ArgumentParser(prog="lpm-fracture")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog", help="List archived run identifiers")
    sub.add_parser("validate", help="Check files and numerical consistency")
    for name in ("reproduce", "plot"):
        command = sub.add_parser(name)
        command.add_argument("--output", default="outputs/" + name)
    command = sub.add_parser("export")
    command.add_argument("--config", required=True)
    command.add_argument("--output", required=True)
    command = sub.add_parser("solve")
    for name in ("config", "solver", "model", "output"):
        command.add_argument("--" + name, required=True)
    command.add_argument("--threads", type=int, default=2)
    command.add_argument(
        "--check", action="store_true", help="Finite-difference energy check only"
    )
    args = parser.parse_args(argv)
    if args.command == "catalog":
        from .data import Archive

        for run in Archive().runs:
            print(run["id"])
    elif args.command == "reproduce":
        from .analysis import reproduce

        result = reproduce(args.output)
        print(
            json.dumps(
                {
                    "checks_passed": result["checks_passed"],
                    "checked": len(result["checks"]),
                    "coarse_2d_means": result["coarse_2d_boundary_fraction_means"],
                },
                indent=2,
            )
        )
    elif args.command == "plot":
        from .plotting import plot_results

        print(json.dumps(plot_results(args.output), indent=2))
    elif args.command == "export":
        from .models import export_model

        print(export_model(args.config, args.output))
    elif args.command == "solve":
        from .solver import run_solver

        run_solver(
            args.solver, args.model, args.output, args.config, args.threads, args.check
        )
    elif args.command == "validate":
        from .validation import validate

        print(json.dumps(validate(), indent=2))
