# ADC Fidelity Benchmark

**How few repeated diffusion acquisitions are enough for a trustworthy apparent diffusion coefficient (ADC) measurement?**

This project benchmarks quantitative fidelity: whether methods preserve known focal ADC changes, and whether their uncertainty identifies failures. Visually clean maps or agreement with a noisy reference are insufficient evidence of accuracy.

## Current status

The first CPU pilot implements original geometric phantoms, signal-level partial volume, complex noise before magnitude, conventional ADC estimation, anatomy-group splits, signed focal-change evaluation, and reproducible tables/plots. It compares magnitude averaging plus log-linear fitting with repetition-level Rician maximum likelihood using known simulated noise scale.

The pilot is a numerical benchmark, with two held-out toy anatomies. Published denoising comparators, anatomical-resource experiments, learned estimation, calibrated uncertainty, and adaptive stopping remain to be implemented. No patient or external anatomical datasets have been downloaded. See [pilot results and limitations](docs/PILOT_RESULTS.md).

The confirmed repository name is `CB9683/adc-fidelity-bench`. The code uses the [MIT license](LICENSE); dataset and other asset permissions remain separate. [Citation metadata](CITATION.cff) initially identifies the owner by the GitHub handle `CB9683`; fuller authorship and release metadata can be added before a research release.

## Scientific design

1. Controlled simulations with known parameters and focal abnormalities; start with `S(b) = S0 exp(-bD)`.
2. Partial volume applied to diffusion signals, and complex noise before magnitude formation.
3. Quantitative comparisons of averaging plus fitting, noise-aware fitting, published denoising, and later a small learned method.
4. ADC error, focal-change recovery, uncertainty calibration, false confidence, and repetitions saved at matched error risk.
5. Optional fastMRI Prostate retrospective validation after access approval, with independent input/reference repetitions and a noisy-reference interpretation.

See the [charter](docs/PROJECT_CHARTER.md), [benchmark specification](docs/BENCHMARK_SPEC.md), [implementation plan](docs/plans/2026-10-02-trustworthy-adc.md), [decisions](docs/DECISIONS.md), and [resource review](docs/RESOURCE_REVIEW.md).

## Run the CPU pilot

Use Python 3.11 or later. The dependency lock records the tested numerical, plotting, test, and build packages; it excludes unrelated packages. Create a local environment from the repository root:

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-build-isolation -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m adc_fidelity_bench benchmark --config configs/pilot.json --output outputs/pilot
```

The equivalent installed command is `adc-fidelity-bench benchmark --config configs/pilot.json --output outputs/pilot`. Installation needs package-index access. The pilot runs on a CPU and creates a new output directory; choose another name for each run because existing results are preserved.

`configs/pilot.json` is executable. `configs/simulation.example.json` remains a broader, incomplete protocol template. Repetition budgets `[1, 2, 4, 8]` are **per b-value**, so two b-values require `[2, 4, 8, 16]` total acquisitions. Reference SNR uses amplitude 1 and component noise sigma; it differs from local tissue SNR.

Outputs include the exact configuration, anatomy assignments, deterministic seeds, source hashes and software versions, `cases.csv`, `anatomy_summary.csv`, `summary.csv`, three descriptive plots, and `example.npz` with assigned fine-resolution parameters and signal-derived coarse references. Noise realizations are averaged within anatomy before giving independent anatomies equal weight. Numerical errors use finite estimates; invalid fractions and defined-case counts expose their denominators. A failed focal voxel suppresses full-region recovery metrics. Inspect failures alongside every error/recovery curve.

The default paired cases share complex noise to isolate changes with lower variance. Set `paired_noise` to `independent` for sensitivity analysis. Neither mode represents measured longitudinal patient data. Rician fits can have an unbounded ADC optimum at the magnitude noise floor, or a zero-amplitude optimum that leaves ADC unidentified. Those ADC values are invalid even if the numerical optimizer converged. Both policies are tested explicitly.

## Repository layout

```text
src/adc_fidelity_bench/  Simulation, estimators, splits, metrics, CLI, reporting
configs/                Executable pilot and broader protocol template
tests/                  Numerical, integration, and data-boundary tests
docs/                   Protocol, data boundaries, sources, and plans
pyproject.toml          Package/build specification
requirements-lock.txt   Scoped exact dependency versions
LICENSE                 MIT code license
CITATION.cff            Initial software citation metadata
```

Real data, anatomical downloads, local manifests, checkpoints, and generated run outputs belong outside Git. Original toy-pilot summary figures may accompany reviewed documentation. See [data layout](docs/DATA_LAYOUT.md).

## Reproducibility and claims

Keep each anatomy and all its noise realizations, perturbations, slices, and acquisition variants in one split. Separate training, model selection, calibration, and final testing. Record configurations, manifests, seeds, versions, and exclusions.

The initial evidence is numerical simulation; later patient work would measure retrospective agreement. This pilot does not establish clinical validity, prospective scanner-time savings, brain-tumour performance, or biological interpretation. It supplies no calibrated error bound and cannot decide when acquisitions are sufficient. Before release, inspect staged content and complete Git history and resolve asset permissions.
