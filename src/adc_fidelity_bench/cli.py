"""Command-line entry point for reproducible simulation pilots."""

import argparse
import json
from pathlib import Path

from .pipeline import run_benchmark


def main(argv=None):
    parser = argparse.ArgumentParser(prog='adc-fidelity-bench')
    commands = parser.add_subparsers(dest='command', required=True)
    benchmark = commands.add_parser('benchmark', help='Run a controlled CPU simulation')
    benchmark.add_argument('--config', required=True, type=Path)
    benchmark.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        with args.config.open() as stream:
            config = json.load(stream)
        result = run_benchmark(config, args.output)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f'error: {error}\n')
    n_anatomies = result['manifest']['n_evaluated_anatomies']
    noun = 'anatomy' if n_anatomies == 1 else 'anatomies'
    print(f"Completed {result['manifest']['n_cases']} cases across {n_anatomies} independent {noun}: {args.output}")
    return 0
