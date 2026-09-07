from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .autotune import run_batch_autotune
from .config import apply_overrides, load_with_overlays, resolve_runtime_paths, save_config
from .evaluation import aggregate_evaluations, configured_benchmark_names, evaluate_suite
from .plotting import plot_results, plot_training_progress
from .preflight import run_preflight
from .trainer import run_training


def _configured(args):
    return resolve_runtime_paths(
        apply_overrides(
            load_with_overlays(args.config, getattr(args, "overlay", [])),
            getattr(args, "overrides", []),
        )
    )


def _visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip():
        return len([item for item in visible.split(",") if item.strip()])
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 1


def _evaluate_checkpoint(args) -> dict:
    config = _configured(args)
    if str(config.get("evaluation", {}).get("backend", "hf")).lower() != "vllm":
        return evaluate_suite(args.name, args.model, config, args.output)
    requested_world = int(os.environ.get("EVAL_WORLD_SIZE", _visible_gpu_count()))
    if requested_world <= 1:
        return evaluate_suite(args.name, args.model, config, args.output)
    from .vllm_evaluation import evaluate_vllm_distributed

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / ".resolved_eval_config.yaml"
    save_config(config, resolved)
    evaluation = config["evaluation"]
    settings = {
        "backend": "vllm",
        "temperature": evaluation.get("temperature", 0.7),
        "top_p": evaluation.get("top_p", 0.95),
        "num_responses": evaluation.get("num_responses", 16),
        "metric": evaluation.get("metric"),
        "max_new_tokens": evaluation.get("max_new_tokens", 2048),
        "limit": evaluation.get("limit"),
        "benchmark_names": list(configured_benchmark_names(config)),
        "vllm": evaluation.get("vllm", {}),
    }
    try:
        return evaluate_vllm_distributed(
            args.name,
            args.model,
            config,
            output,
            settings,
            resolved,
            world_size=requested_world,
        )
    finally:
        resolved.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone B200 OPD, TA-OPD, Bellman-RAC, and PGT experiment"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--overlay", action="append", default=[])
    train.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE"
    )

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--overlay", action="append", default=[])
    preflight.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE"
    )
    preflight.add_argument("--output", required=True)

    tune = commands.add_parser("autotune-batch")
    tune.add_argument("--opd-config")
    tune.add_argument("--ta-config", required=True)
    tune.add_argument("--rac-config", required=True)
    tune.add_argument("--output", required=True)
    tune.add_argument("--generated-config", required=True)
    tune.add_argument("--candidates", nargs="+", type=int)
    tune.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE"
    )

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--overlay", action="append", default=[])
    evaluate.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE"
    )
    evaluate.add_argument(
        "--name",
        required=True,
        choices=("Base", "OPD", "TA-OPD", "RAC", "PGT", "CMT-OPD"),
    )
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--output", required=True)

    aggregate = commands.add_parser("aggregate-eval")
    aggregate.add_argument("--base-dir", required=True)
    aggregate.add_argument("--opd-dir")
    aggregate.add_argument("--ta-dir", required=True)
    aggregate.add_argument("--rac-dir", required=True)
    aggregate.add_argument("--pgt-dir")
    aggregate.add_argument("--cmt-dir")
    aggregate.add_argument("--output", required=True)

    plot = commands.add_parser("plot")
    plot.add_argument("--results", required=True)
    plot.add_argument("--opd-output")
    plot.add_argument("--ta-output", required=True)
    plot.add_argument("--rac-output", required=True)
    plot.add_argument("--pgt-output")
    plot.add_argument("--cmt-output")
    plot.add_argument("--smoothing-window", type=int, default=10)
    plot.add_argument("--plot-name")

    progress_plot = commands.add_parser("plot-training-progress")
    progress_plot.add_argument("--results", required=True)
    progress_plot.add_argument(
        "--method",
        default="both",
        choices=(
            "all",
            "both",
            "opd",
            "pure-opd",
            "ta",
            "ta-opd",
            "rac",
            "bellman-rac",
            "pgt",
            "cmt",
        ),
        help="Legacy single selector: all, both=TA+RAC, or one method",
    )
    progress_plot.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "opd", "pure-opd", "ta", "ta-opd", "rac", "bellman-rac", "pgt", "cmt"
        ),
        help="One or more methods to plot in the requested order",
    )
    progress_plot.add_argument("--opd-output")
    progress_plot.add_argument("--ta-output")
    progress_plot.add_argument("--rac-output")
    progress_plot.add_argument("--pgt-output")
    progress_plot.add_argument("--cmt-output")
    progress_plot.add_argument("--smoothing-window", type=int, default=10)
    progress_plot.add_argument("--plot-name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        run_training(_configured(args), command_line=sys.argv)
        return 0
    if args.command == "preflight":
        result = run_preflight(_configured(args), args.output)
    elif args.command == "autotune-batch":
        result = run_batch_autotune(
            args.ta_config,
            args.rac_config,
            args.output,
            args.generated_config,
            args.candidates,
            opd_config=args.opd_config,
            overrides=args.overrides,
        )
    elif args.command == "evaluate":
        result = _evaluate_checkpoint(args)
    elif args.command == "aggregate-eval":
        model_dirs = {"Base": args.base_dir}
        if args.opd_dir:
            model_dirs["OPD"] = args.opd_dir
        model_dirs.update({"TA-OPD": args.ta_dir, "RAC": args.rac_dir})
        if args.pgt_dir:
            model_dirs["PGT"] = args.pgt_dir
        if args.cmt_dir:
            model_dirs["CMT-OPD"] = args.cmt_dir
        result = aggregate_evaluations(
            model_dirs,
            args.output,
        )
    elif args.command == "plot":
        result = plot_results(
            args.results,
            args.ta_output,
            args.rac_output,
            args.smoothing_window,
            args.plot_name,
            opd_output=args.opd_output,
            pgt_output=args.pgt_output,
            cmt_output=args.cmt_output,
        )
    elif args.command == "plot-training-progress":
        result = plot_training_progress(
            args.results,
            args.ta_output,
            args.rac_output,
            args.smoothing_window,
            args.plot_name,
            method=args.method,
            opd_output=args.opd_output,
            methods=args.methods,
            pgt_output=args.pgt_output,
            cmt_output=args.cmt_output,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
