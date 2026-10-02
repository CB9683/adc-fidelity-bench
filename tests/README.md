# Scientific verification

Create a local environment and run the test suite from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation -e '.[dev]'
python -m pytest
```

The lock records the tested numerical/test dependencies and build tooling for Python 3.11+ on macOS/Linux. It excludes unrelated installed packages. GitHub CI is configured to check Python 3.11 and 3.13; only the locally executed versions may be claimed as locally verified. A pinned version file does not verify downloaded artifact hashes.

Known-answer and adversarial tests cover monoexponential units and zero-noise recovery, signal-level mixtures differing from averaged tissue ADC, complex Gaussian noise before magnitude formation, seeded geometry and repetition axes, conventional ADC fits and optimizer failures, whole-anatomy split isolation, and signed focal-change recovery with invalid-estimate counts. Repository boundary checks keep private data excluded while allowing the source-code data package.

The generated phantom is original toy geometry, and these checks do not establish anatomical realism or clinical performance. Later learned estimation, disjoint patient-data input/reference repeats, calibrated interval coverage, sequential stopping, and repetition savings require their own scientific tests before implementation claims.
