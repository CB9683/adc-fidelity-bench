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

The learning suite additionally covers grouped new train/validation/calibration/test IDs, held-out shape/configuration validation, observed-only normalization and repetition prefixes, signal-derived targets, NaN-safe per-image masked loss and gradients, physical output scaling, actual optimizer learning, validation-only selection, restricted checkpoint reload, matched evaluation inputs, configured recovery thresholds and optional-runtime CLI behavior. All 223 tests pass with the learning runtime in the tested local environment. Neural tests skip when optional PyTorch is absent; the baseline CLI remains usable.

For the learning environment recipe and exact lock scope, see [FIRST_CNN_METHOD.md](../docs/FIRST_CNN_METHOD.md). A separate Linux/Python 3.13 CI job installs CPU PyTorch, runs the expanded suite and exercises the installed training CLI on a tiny independent cohort. CI configuration is distinct from observed CI success.

The generated phantom is original toy geometry, and these checks do not establish anatomical realism or clinical performance. Disjoint patient-data input/reference repeats, calibrated interval coverage, sequential stopping and repetition savings require their own scientific tests before implementation claims.
