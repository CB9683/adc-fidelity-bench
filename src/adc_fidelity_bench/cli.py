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
    learning = commands.add_parser('train-cnn', help='Train and evaluate the first small ADC CNN')
    learning.add_argument('--config', required=True, type=Path)
    learning.add_argument('--output', required=True, type=Path)
    ablation = commands.add_parser('ablate-cnn', help='Compare matched ADC models and focal-preservation objectives')
    ablation.add_argument('--config', required=True, type=Path)
    ablation.add_argument('--output', required=True, type=Path)
    diagnostic = commands.add_parser('diagnose-cnn', help='Probe a frozen first CNN on its validation geometries')
    diagnostic.add_argument('--reference-run', required=True, type=Path)
    diagnostic.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'diagnose-cnn':
            try:
                from .learning.diagnostics import run_first_cnn_diagnostics
            except ModuleNotFoundError as error:
                if error.name != 'torch':
                    raise
                parser.exit(2, 'error: PyTorch is optional; install the train extra or requirements-learning-lock.txt\n')
            result = run_first_cnn_diagnostics(args.reference_run, args.output)
        else:
            with args.config.open() as stream:
                config = json.load(stream)
            if args.command in ('train-cnn', 'ablate-cnn'):
                try:
                    if args.command == 'train-cnn':
                        from .learning.experiment import run_cnn_experiment as run_learning
                    else:
                        from .learning.ablation import run_ablation as run_learning
                except ModuleNotFoundError as error:
                    if error.name != 'torch':
                        raise
                    parser.exit(2, 'error: PyTorch is optional; install the train extra or requirements-learning-lock.txt\n')
                result = run_learning(config, args.output)
            else:
                result = run_benchmark(config, args.output)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f'error: {error}\n')
    n_anatomies = result['manifest']['n_evaluated_anatomies']
    noun = 'anatomy' if n_anatomies == 1 else 'anatomies'
    print(f"Completed {result['manifest']['n_cases']} cases across {n_anatomies} independent {noun}: {args.output}")
    return 0
