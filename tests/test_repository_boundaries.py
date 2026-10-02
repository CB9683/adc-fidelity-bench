from pathlib import Path
import subprocess


def test_source_data_handling_code_can_be_versioned():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['git', '-C', str(root), '-c', 'core.excludesFile=/dev/null',
                             'check-ignore', '--no-index', '-q',
                             'src/adc_fidelity_bench/data/splits.py'])
    assert result.returncode == 1


def test_private_dataset_and_generated_output_remain_excluded():
    root = Path(__file__).resolve().parents[1]
    for path in ('data/participant.csv', 'outputs/pilot/cases.csv', 'scan.nii.gz',
                 'CODEX_STARTER_PROMPT.md'):
        result = subprocess.run(['git', '-C', str(root), '-c', 'core.excludesFile=/dev/null',
                                 'check-ignore', '--no-index', '-q', path])
        assert result.returncode == 0, path
