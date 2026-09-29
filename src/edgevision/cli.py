"""Usage: edgevision infer --config configs/inference.yaml."""

import argparse
import json
from pathlib import Path

from edgevision.config import load_config
from edgevision.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline local-video detection on CPU")
    subcommands = parser.add_subparsers(dest="command", required=True)
    infer = subcommands.add_parser("infer", help="Process each video frame into JSONL results")
    infer.add_argument("--config", required=True, type=Path)
    infer.add_argument(
        "--output-dir", type=Path, help="New run directory, relative to current directory"
    )
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.output_dir is not None:
            config = config.model_copy(update={"output_dir": args.output_dir.resolve()})
        result = run(config)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"edgevision: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
