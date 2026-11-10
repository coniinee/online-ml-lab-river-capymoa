"""Run Part 2 in its isolated environment and one fresh process per model run."""

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from capy_lab.runtime import configure
from scripts.redact_public_paths import redact_file, redact_distribution
CONFIG=configure()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["worker","classification","clustering","automl","all","analyze"])
    parser.add_argument("--job",type=Path)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.command=="analyze":
        from capy_lab.analysis import analyze
        analyze();return
    from capy_lab.experiments import run,worker
    if args.command=="worker":
        job=json.loads(args.job.read_text())
        worker(job,CONFIG)
        redact_file(ROOT/'results/capymoa/runs'/f"{job['id']}.json")
    else:
        run(CONFIG,["classification","clustering","automl"] if args.command=="all" else [args.command],args.resume)
        redact_distribution()


if __name__=="__main__":main()
