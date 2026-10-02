# Focal Preservation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute a controlled spatial-context × training-objective comparison with matched capacity, fresh geometry groups, multiple noise draws and zero-change controls, and report the joint quantitative trade-off.

**Architecture:** Add separate ablation data/model/runner modules, reusing the tested complex-noise simulator, signal-derived references, observed preprocessing and conventional estimators. Preserve the original CNN experiment. Split new geometry IDs before all derivatives and keep the final cohort untouched until source/configuration/checkpoints are frozen.

**Tech Stack:** Existing locked Python/PyTorch CPU environment, NumPy/SciPy, Matplotlib Agg, pytest, JSON/CSV and restricted tensor checkpoints. Use an isolated temporary worktree and independent implementation/review scopes.

---

## Prespecified scientific contract

- New IDs `abl-000`…`abl-143`, seed 2026100202, split 64 train / 16 validation / 32 reserved calibration / 32 final test; exclude all old `toy-`/`cnn-` IDs. Calibration acquisition examples/scores remain unused. Configuration may validate noiseless geometry validity across roles.
- Original 16×16, 0.5 mm fine grid and 2×2 signal averaging to 8×8, 1 mm acquisition grid; b=[0,1000] s/mm², R=4 per b, eight total acquisitions; SNR=[5,20]. Every method sees the same measurements. Only Rician MLE receives known simulated sigma.
- Training/validation: varied tissue ADC/S0 and valid random focal locations, circles only. Inner ADC uniform [0.0007,0.0012], outer [0.0013,0.0019] mm²/s; inner S0 [0.65,0.95], outer [0.85,1.15]. These are numerical assumptions, not biological ranges. Radius=[0.5,1.5] mm, contrast=[-0.0003,-0.00015,0,0.00015,0.0003] mm²/s, two train noise draws, two validation draws. Geometry/tissue/location choices are deterministic and fixed across contrast signs, SNR/noise draws and all model cells; sampling position may depend on size/shape/regime to ensure support. Reuse a noise draw across contrast values within a condition, recording that scope and clustering within geometry.
- Final test: original-like fixed-parameter/central-focus control and varied regime, circles/ellipses, same SNR/radius/contrast grid, three noise draws, independent noise primary and common noise sensitivity. Ellipses are absent from train/validation; nominal equal-area aspect ratio 2.25. Paired zero-change observations have identical noiseless signals but independent noise in the primary pairing. All conditions remain clustered within geometry.
- Four cells: spatial(2→16→16→1, 3×3/3×3/1×1, 2641 params) or pointwise(2→49→49→1, all1×1, 2647 params), crossed with ordinary tissue MSE or tissue MSE + 1.0×paired focal-mean change MSE. ReLU after hidden layers, Softplus×1e-3 ADC output. Retain observed b0-p95 normalization in every cell; pointwise means no learned spatial mixing, not wholly independent preprocessing.
- Inputs remain two observed magnitude means only. Supervised ADC/true tissue/focal masks and paired references enter losses/evaluation only. Predict each member separately through the same network; never pass paired truth/location/masks to inference.
- Pair loss: average per-image scaled-ADC tissue MSE across both images, plus weight×mean across pairs of squared error in the signed focal-ROI mean ADC difference scaled by 1000. Include δ=0 pairs. Select valid voxels before subtraction to avoid NaN gradients. Define focal mean over the full coarse focal footprint.
- Adam lr=.001, batch64 **pairs**, 80 epochs, CPU2 threads/deterministic operations, seeds11/22/33. Same data and per-seed shuffle sequence in every cell. For all cells select earliest minimum validation score `tissue MSE + 1.0×change MSE`; training objective alone differs. Selection weights and settings are frozen here, with no test optimization. Report both validation components and all seeds, not a test-selected winner.
- Frozen-first-CNN diagnosis: validate checkpoint/config hashes; use its original validation IDs only, circles, both sizes/signs, observed normalization, noiseless and one SNR100 draw per condition with independent healthy/perturbed noise. Include averaging/log fitting as the zero-noise reference sanity control. Report ADC error and signed recovery; this distributional diagnostic does not prove the cause of attenuation. Do not inspect reserved calibration/final-test groups for diagnosis or debugging.
- Reuse full paired ROI metrics; record finite-only errors with requested/valid/failure counts. Add absolute focal change error, opposite-sign recovery, finite max tissue error, zero-change absolute focal change and false-change indicator at prespecified |δhat|>1e-4 mm²/s. Report null alarm-or-invalid rate over all requested null pairs to avoid rewarding missing predictions. Recovery is undefined for δ=0 and is not a detection sensitivity.
- Aggregate noise within geometry/condition, then geometries equally; preserve regime and null/change distinctions, and all undefined counts. Compact tables average conditions within geometry before cohort means. Optimization seeds are sensitivity replicates, not independent anatomies. Plot ADC error versus focal recovery alongside null/spillover errors and invalid fractions; no guaranteed improvement or calibrated risk claim.
- Richer-data effects are not identifiable by comparison to the historical first CNN. The primary four-cell effects are measured within the same new data regime; the original-like test is a separate control. No uncertainty/stopping, clinical, acquisition-reduction or publication-novelty claim.

## Task 1: Paired ablation data

**Create:** `learning/ablation_data.py`, `tests/test_ablation_data.py`, `configs/focal_ablation.json`.

1. Write/run failing known-answer and boundary tests for fresh disjoint groups, deterministic varied ADC/S0/location, original-like equivalence, δ=0 signals/references, independent/common noise, target construction, evaluation regimes/shapes and invalid configurations.
2. Implement `ablation_splits(config)` and `iter_ablation_cases(config, split, *, regimes=None, focal_shapes=None, paired_noise='independent', noise_realizations=None)` returning existing `LearningCase`. Metadata adds regime, case_kind, focal center and tissue parameters. Do not import torch.
3. Validate all registered geometry and numerical settings before allocating training output. Keep original modules unchanged. Test and independently review.

## Task 2: Matched models and paired loss

**Create:** `learning/ablation_model.py`, `tests/test_ablation_model.py`.

1. Write/run failing tests for exact parameter counts, unchanged spatial architecture, no pointwise learned neighbour dependence, units/shape, safe checkpoints and NaN-safe equal-image paired loss.
2. Implement `make_ablation_model(architecture)` with spatial/pointwise choices; `paired_loss_components(healthy_prediction, prediction, healthy_target, target, tissue_mask, focal_mask)` returns tensors `mse` and `change_mse` with finite values/gradients and strict shapes/full support.
3. Loss weights are applied by the runner; masks never enter observed preprocessing/inference. Verify null and signed contrasts, invalid/empty masks, analytic loss and gradients. Independently review.

## Task 3: Diagnostics, runner and reporting

**Create:** `learning/ablation.py`, `learning/diagnostics.py`, corresponding tests.
**Modify:** CLI and learning CI only.

1. Write/run failing integration tests for genuine optimization, common validation selection, model specification/reload, same-pair evaluation, δ=0 denominators, regime-preserving aggregation, provenance and preserved output.
2. Add lazy `ablate-cnn` and `diagnose-cnn` installed commands. Preserve baseline optional-runtime behavior. Train all 12 cells/seeds on one shared paired dataset; save histories/registry/config/source/checkpoint hashes and failed status on errors. Reload every selected checkpoint.
3. Evaluate final data only after all checkpoints are frozen, caching/batching only identical observed inputs without altering metrics. Include both conventional methods. Record component losses, elapsed times, device/versions and matched measurements.
4. Create actual joint/error/recovery/null/invalid figures with file-only Agg. Save raw and geometry/condition summaries and reproducible example maps. Inspect plots and denominator arithmetic.
5. Run focused/full suites, core without torch, build/install/wheel/CLI smoke and independent spec/code reviews. Fix before the research run. Smoke uses a separate tiny cohort, never the final test.

## Task 4: Execute and integrate

1. Run frozen-first-CNN development diagnostics; save measured output/limitations.
2. Freeze clean source/config and run the prespecified four-cell experiment, including fresh final evaluation. Do not alter architecture/objective/config after final-test inspection.
3. Independently audit hashes, epoch selection, grouping, noise matching, metrics, failure/null denominators and saved example predictions. Report all seeds, sign/size/shape/regime/pairing effects and trade-offs; do not infer mechanism solely from one comparison.
4. Write `docs/FOCAL_PRESERVATION_METHOD.md` and `docs/FOCAL_PRESERVATION_RESULTS.md`, update implemented status, and retain reviewed original summary figures only. Keep checkpoints, arrays and raw run outputs ignored.
5. Run verification/privacy audit, merge to main, update the already authorized repository and verify CI. Preserve local research artifacts and working environment; remove the completed temporary worktree after integration. User authorized the proposed experiment; no repeated routine permission question is required.
