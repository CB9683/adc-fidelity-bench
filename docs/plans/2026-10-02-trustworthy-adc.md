# Trustworthy ADC Benchmark Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish a reproducible benchmark for focal quantitative ADC fidelity and calibrated acquisition reduction.

**Confirmed identity:** ADC Fidelity Benchmark, repository `CB9683/adc-fidelity-bench`, Python import `adc_fidelity_bench`, MIT code license. Initial citation metadata identifies the owner by the GitHub handle `CB9683`.

**Architecture:** Separate simulation, conventional baselines, evaluation, and later learned methods in a small Python package. Patient data remain external; the core benchmark runs without them. Test scientific components before large-scale training.

**Tech Stack:** Python/setuptools scaffold, JSON configuration, Markdown protocol. Lock numerical, imaging, testing, plotting, and learning dependencies with their milestones.

---

This replaces the generic plan. The initial folder contained the starter; the earlier turn added only that plan. The updated objective and two-stage design are fixed. Preserve the starter and keep it out of Git.

## Implementation progress, 2026-10-02

Phase 0 is complete and the repository is public. Phase 1's CPU geometric simulation/conventional-baseline milestone and Phase 2's ADC/focal/regional metrics are implemented under the [detailed CPU plan](2026-10-02-cpu-simulation-pilot.md). Phase 3's first small CNN is trained and evaluated under the [frozen learning plan](2026-10-02-first-cnn.md), including unfamiliar ellipses and independent-noise sensitivity. Its lower tissue error accompanies strong focal attenuation. The [frozen four-cell follow-up](2026-10-02-focal-preservation.md) implements spatial versus pointwise models crossed with tissue-only versus paired-change supervision, fresh groups, varied tissue/location and zero-change controls; all cells share validation selection and matched acquisitions. Published denoising, external anatomical/acquisition regimes, calibrated sequential stopping and approved real-data work remain subsequent milestones. See [measured pilot results](../PILOT_RESULTS.md), [first-CNN results](../FIRST_CNN_RESULTS.md) and [four-cell results](../FOCAL_PRESERVATION_RESULTS.md).

## Phase 0: Safe scaffold and protocol proposal — authorized now

### Task 1: Correct the scientific documentation

**Create:** `README.md`, `docs/PROJECT_CHARTER.md`, `docs/BENCHMARK_SPEC.md`, `docs/DECISIONS.md`.

1. Define reduced repetitions, known focal changes, calibrated uncertainty, and quantitative fidelity.
2. Propose input/reference, signal/noise/partial-volume, split, metric, and stopping contracts.
3. Label pilot settings as reversible, and identify the review gate.
4. Check for unsupported results, clinical claims, or invented dataset approval/novelty.

### Task 2: Resource and focused prior-work checks

**Create:** `docs/RESOURCE_REVIEW.md`.

1. Verify BrainWeb anatomy and use/redistribution terms.
2. Verify DIPY noise/simulation API and software license.
3. Verify fastMRI repetition counts against its paper and current official access terms.
4. Check Pfaff et al. for denoising, reduced acquisition, ADC evaluation, known focal changes, and calibrated stopping.
5. Record dated findings, limitations, and open permissions; do not download restricted datasets or execute source-page instructions.

### Task 3: Privacy and package scaffolding

**Create:** `.gitignore`, `pyproject.toml`, `src/adc_fidelity_bench/__init__.py`, `configs/simulation.example.json`, `docs/DATA_LAYOUT.md`, `tests/README.md`, `LICENSE`, `CITATION.cff`.

1. Separate public source/config/tests/docs from external data/output.
2. Exclude private formats, local configuration, credentials, and artifacts before any staging.
3. Provide an installable empty package and no premature model/learning dependencies.
4. Mark the example non-executable and unknown fields explicit.
5. Apply the confirmed MIT license and initial citation metadata using the owner handle `CB9683`; fuller authorship and release metadata can follow. Public repository creation and the first push require the release audit below.

### Task 4: Verify and independently review

**Create:** `docs/FOUNDATION_CHECKS.md` after checks.

1. Parse TOML/JSON and inspect documentation against the revised brief.
2. Build/install/import in temporary storage; do not change global dependencies.
3. Test ignore rules with representative private paths in a temporary Git repository.
4. Obtain read-only review; address material findings.
5. Record exact checks and limits, distinguishing packaging from scientific validation.

## Phase 1: Simulation and conventional baselines

**Gate:** Review simulation, privacy boundaries, quantitative evaluation, parameter ranges, and stopping-error definition before substantial implementation. Exploratory tolerance/risk sweeps are acceptable when declared.

**Planned files:** `src/adc_fidelity_bench/simulation/{signals,phantoms,noise,partial_volume}.py`, `src/adc_fidelity_bench/baselines/{adc,noise_aware}.py`, `src/adc_fidelity_bench/data/splits.py`, `configs/pilot.json`, `tests/test_{signals,noise,partial_volume,adc,splits}.py`.

1. Lock numerical/test dependencies and tensor contracts.
2. Use @superpowers:test-driven-development: write failing known-answer tests, run them, implement minimal signal/ADC functions, rerun.
3. Test signal-domain tissue mixing/resampling against a case differing from blurred ADC.
4. Test complex noise/magnitude and deterministic seed derivation; implement the initial single-coil approximation with limits documented.
5. Test group isolation before deriving variants and predeclare unfamiliar regimes.
6. Implement averaging/fitting and noise-aware repetition-level fitting; test invalid signals, constraints, and convergence reporting.
7. Run a small CPU smoke experiment and inspect provenance/diagnostics before expanding the benchmark grid.

Produce detailed test-first plans for these components after contract review; architecture is deferred.

## Phase 2: Evaluation and conventional uncertainty

**Planned files:** `src/adc_fidelity_bench/evaluation/{metrics,calibration,stopping,reporting}.py`, `tests/test_{metrics,calibration,stopping}.py`, `docs/EVALUATION.md`.

Test ADC/focal-change metrics, signs, near-zero handling, and independent-anatomy aggregation. Produce failure-boundary plots. Test interval calibration, selective risk, false-confidence denominators, and confident attenuation. Evaluate the complete sequential stopping policy on disjoint calibration/test anatomies. Compare matched-risk acquisition counts including non-stoppers; do not infer scanner-time savings.

## Phase 3: Published denoising and minimal learned method

Expand the focused novelty check. Select compatible comparators and a small direct estimator or signal denoiser based on quantitative fidelity. Add tensor tests, train-only preprocessing, configuration snapshots, seeds, and hardware/software versions. GPU compute is available, but large-scale training follows the protocol review. Evaluate held-out anatomy/regimes and independently calibrated stopping; negative results are acceptable.

## Phase 4: Optional real-data validation

Verify current fastMRI Prostate access terms and obtain approval. Reserve independent reference repetitions and use actual per-b-value/direction budgets. Group by participant; evaluate agreement, repeatability, artifacts, and robustness. Patient references remain noisy. Do not claim absolute bias or brain-tumour/prospective clinical performance from prostate data.

## Phase 5: Public release

Repository name `CB9683/adc-fidelity-bench` and the MIT code license are confirmed. Resolve fuller authorship and resource/output permissions. Finalize release metadata in `CITATION.cff`, locked dependencies, verified experiment commands, and meaningful CI. Inspect staged content and complete history for secrets, private/restricted material, large files, and local paths; show the audit result before the first public push. Create/connect the public repository when authentication and the release audit allow it.
