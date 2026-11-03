#!/usr/bin/env python3
"""Prepare a masked packet, print a blank template, or retain a human submission."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from persona_observatory.calibration import prepare_packet, submission_template, submit
from persona_observatory.storage import json_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--archive', type=Path, required=True)
    prepare.add_argument('--plan', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    template = sub.add_parser('template')
    template.add_argument('--participant', type=Path, required=True)
    template.add_argument('--batch', required=True)
    accept = sub.add_parser('submit')
    accept.add_argument('--participant', type=Path, required=True)
    accept.add_argument('--submission', type=Path, required=True)
    accept.add_argument('--records', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare_packet(args.archive, args.plan, args.output)
    elif args.command == 'template':
        result = submission_template(args.participant, args.batch)
    else:
        result = submit(args.participant, args.submission, args.records)
    print(json_text(result), end='')
    return int(result.get('status') == 'REJECTED')


if __name__ == '__main__':
    raise SystemExit(main())
