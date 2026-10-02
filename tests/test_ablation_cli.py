import subprocess
import sys


def test_help_exposes_ablation_and_development_diagnosis_without_runtime_import():
    completed=subprocess.run([sys.executable,'-m','adc_fidelity_bench','--help'],capture_output=True,text=True)
    assert completed.returncode==0
    assert 'ablate-cnn' in completed.stdout and 'diagnose-cnn' in completed.stdout
    unloaded=subprocess.run([sys.executable,'-c','import sys; from adc_fidelity_bench import cli; assert "torch" not in sys.modules'],capture_output=True,text=True)
    assert unloaded.returncode==0,unloaded.stderr
