# ADC Fidelity Benchmark

**How few repeated diffusion acquisitions are enough for a trustworthy apparent diffusion coefficient (ADC) measurement?**

This project will benchmark quantitative fidelity: whether methods preserve known focal ADC changes, and whether their uncertainty identifies failures. Visually clean maps or agreement with a noisy reference are insufficient evidence of accuracy.

## Current status

Initial project scaffold. There is no simulation engine, scientific estimator, learned model, stopping rule, or research result yet. The proposed benchmark and evaluation plan require review before substantial implementation or large-scale training. No datasets have been downloaded or research results generated.

The confirmed repository name is `CB9683/adc-fidelity-bench`. The code uses the [MIT license](LICENSE); dataset and other asset permissions remain separate. [Citation metadata](CITATION.cff) initially identifies the owner by the GitHub handle `CB9683`; fuller authorship and release metadata can be added before a research release.

## Scientific design

1. Controlled simulations with known parameters and focal abnormalities; start with `S(b) = S0 exp(-bD)`.
2. Partial volume applied to diffusion signals, and complex noise before magnitude formation.
3. Quantitative comparisons of averaging plus fitting, noise-aware fitting, published denoising, and later a small learned method.
4. ADC error, focal-change recovery, uncertainty calibration, false confidence, and repetitions saved at matched error risk.
5. Optional fastMRI Prostate retrospective validation after access approval, with independent input/reference repetitions and a noisy-reference interpretation.

See the [charter](docs/PROJECT_CHARTER.md), [benchmark specification](docs/BENCHMARK_SPEC.md), [implementation plan](docs/plans/2026-10-02-trustworthy-adc.md), [decisions](docs/DECISIONS.md), and [resource review](docs/RESOURCE_REVIEW.md).

## Installation of the scaffold

Use Python 3.11 or later. There are currently no scientific runtime dependencies. Numerical/learning dependencies and a lockfile will accompany the first implementation milestone.

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -c 'import adc_fidelity_bench; print(adc_fidelity_bench.__version__)'
```

The import should report `0.1.0.dev0`. Installation may need package-index access for the build backend. Scientific tests and training/evaluation commands will be added with their implementations.

## Repository layout

```text
src/adc_fidelity_bench/  Python package; scientific modules are forthcoming
configs/                Public versioned configuration templates
tests/                  Test requirements; numerical tests are forthcoming
docs/                   Protocol, data boundaries, sources, and plans
pyproject.toml          Package/build specification
LICENSE                 MIT code license
CITATION.cff            Initial software citation metadata
```

Planned scientific modules are `simulation/`, `baselines/`, `evaluation/`, and later `models/`. Real data, anatomical downloads, local manifests, checkpoints, and generated outputs belong outside Git. See [data layout](docs/DATA_LAYOUT.md).

## Reproducibility and claims

Keep each anatomy and all its noise realizations, perturbations, slices, and acquisition variants in one split. Separate training, model selection, calibration, and final testing. Record configurations, manifests, seeds, versions, and exclusions.

The planned initial evidence is simulation-based and later retrospective agreement. It does not establish clinical validity, prospective scanner-time savings, brain-tumour performance, or biological interpretation. Before release, inspect staged content and complete Git history and resolve asset permissions.
