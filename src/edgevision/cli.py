"""Usage: edgevision infer --config configs/inference.yaml."""

import argparse
import json
from pathlib import Path

from edgevision.config import load_config
from edgevision.data_audit import SPLITS, audit_image
from edgevision.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU perception and data inspection")
    subcommands = parser.add_subparsers(dest="command", required=True)
    infer = subcommands.add_parser("infer", help="Process each video frame into JSONL results")
    infer.add_argument("--config", required=True, type=Path)
    infer.add_argument(
        "--output-dir", type=Path, help="New run directory, relative to current directory"
    )
    audit = subcommands.add_parser("audit-image", help="Inspect one VisDrone DET annotation pair")
    audit.add_argument("--image", required=True, type=Path)
    audit.add_argument("--annotations", required=True, type=Path)
    audit.add_argument("--split", required=True, choices=SPLITS)
    audit.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "audit-image":
            result = audit_image(args.image, args.annotations, args.split, args.output_dir)
        else:
            config = load_config(args.config)
            if args.output_dir is not None:
                config = config.model_copy(update={"output_dir": args.output_dir.resolve()})
            result = run(config)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"edgevision: {exc}\n")
    print(json.dumps(result, indent=2))
    if result["status"] == "needs_review":
        parser.exit(1)


if __name__ == "__main__":
    main()
