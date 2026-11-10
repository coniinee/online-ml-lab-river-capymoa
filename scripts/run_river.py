"""CLI: prepare data, run steps 2–5, then build figures and analysis."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "analyze", "all"])
    parser.add_argument("--config", type=Path, default=ROOT / "config/river.json")
    parser.add_argument("--resume", action="store_true", help="Reuse completed run JSON files with matching configuration")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    from river_lab.data import prepare

    if args.command in ("prepare", "all"):
        prepare(config)
    if args.command in ("run", "all"):
        from river_lab.evaluation import run_all
        run_all(config, resume=args.resume)
    if args.command in ("analyze", "all"):
        from river_lab.analysis import analyze
        analyze()


if __name__ == "__main__":
    main()
