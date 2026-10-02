import subprocess
import sys


def test_help_exposes_training_without_importing_torch():
    completed = subprocess.run([sys.executable, '-m', 'adc_fidelity_bench', '--help'],
                               capture_output=True, text=True)
    assert completed.returncode == 0
    assert 'train-cnn' in completed.stdout


def test_cli_help_keeps_optional_torch_runtime_unloaded():
    completed = subprocess.run([sys.executable, '-c',
        'import sys; from adc_fidelity_bench import cli; '
        'assert "torch" not in sys.modules; '
        'print("optional-runtime-unloaded")'], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
