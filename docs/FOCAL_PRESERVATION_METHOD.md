# Focal preservation: frozen four-cell study

This study crosses learned spatial context with the supervised training objective. It follows the [first CNN's measured focal attenuation](FIRST_CNN_RESULTS.md) and keeps the original experiment unchanged. The [executable configuration](../configs/focal_ablation.json) and [implementation plan](plans/2026-10-02-focal-preservation.md) specify all choices before final evaluation. Improvement is a hypothesis, not an assumed outcome.

## Four matched comparisons

| Cell | Architecture | Parameters | Training objective |
| --- | --- | ---: | --- |
| Spatial MSE | Original 2→16→16→1, 3×3/3×3/1×1 | 2,641 | Tissue ADC MSE |
| Pointwise MSE | 2→49→49→1, all 1×1 | 2,647 | Tissue ADC MSE |
| Spatial + change | Same spatial architecture | 2,641 | Tissue MSE + paired focal-change MSE |
| Pointwise + change | Same pointwise architecture | 2,647 | Tissue MSE + paired focal-change MSE |

ReLU follows both hidden layers; Softplus×1e-3 produces ADC in mm²/s. No pooling, batch normalization, perceptual loss or explicit smoothing is added. Pointwise means no learned spatial mixing. Every cell retains the same global observed b0-percentile normalization, so preprocessing is not completely voxel-independent. Matching parameter count does not make architectural expressiveness identical.

Each observation supplies only two magnitude-mean images, using four repetitions per b-value at b=0/1000 s/mm²: eight total acquisitions. Divide both channels by the observed mean-b0 image's 95th percentile, floored at 1e-6. Each member of a pair is predicted separately by the same model. Batching images does not supply their pairing, truth, masks, generator sigma or focal location to inference.

Tissue loss averages per-image MSE of ADC×1000 across both healthy/perturbed images, then pairs equally. Change loss averages squared error in each pair's signed focal-ROI mean ADC difference, also scaled by 1000. The change coefficient is 1.0; it is zero for MSE-only training. Synthetic true masks/paired targets define supervised loss support only. Full focal footprints are used, including partial volume; NaNs outside support are excluded before arithmetic.

All four cells use the **same validation selection score: tissue MSE + 1.0×change MSE**, retaining the earliest minimum on an exact tie. This selection score is used even for MSE-only training; it differs from the historical first experiment's tissue-only selection. Adam uses lr=.001, batch64 **pairs**, 80 epochs, CPU2 threads/deterministic operations and seeds11/22/33. Each cell uses identical paired data and, for a given seed, the same private shuffle sequence. All 12 selected checkpoints are reloaded with `weights_only=True` and reported; final testing selects nothing.

## New groups and controlled signal generation

144 new `abl-` IDs, seed2026100202, are split before derivatives into 64 train, 16 validation, 32 reserved calibration and 32 final-test groups. Old `toy-`/`cnn-` IDs are excluded. Calibration acquisitions/scores remain unused; configuration validation checks noiseless geometry feasibility across roles. These are independent numerical geometry groups, not patient anatomies.

Fine grids remain16×16 at0.5mm, with2×2 signal averaging to8×8 at1mm. Generate monoexponential tissue diffusion signals, average signals spatially, then add independent Gaussian real/imaginary noise before magnitude formation. Fit the reference ADC from noiseless acquisition-resolution signals. Assigned ADC is never blurred to form the target.

Training/validation use **varied circles only**. Inner/outer tissue ADC are uniform over [0.0007,0.0012]/[0.0013,0.0019] mm²/s; S0 over [0.65,0.95]/[0.85,1.15]. These are declared numerical assumptions. Tissue parameters and parent geometry remain fixed by group/regime. Focal centers are sampled uniformly among valid fine-grid centers with the full focus inside tissue; center may vary with radius/shape/regime to retain support. Tissue/location remain fixed across contrasts, SNR and noise draws.

Both radii0.5/1.5mm and contrasts −0.0003/−0.00015/0/+0.00015/+0.0003mm²/s are included, at reference SNR5/20. Training and validation use two noise draws per condition. This produces2,560 training pairs (5,120 images) and640 validation pairs (1,280 images). Noise seeds deliberately exclude contrast: all contrasts/null share a noise draw within a condition. These correlated variants remain clustered within geometry.

The ablation wrapper's training/evaluation draw counts and evaluation pairings supersede the inherited simulation draw/pairing fields. Only the final-test role is evaluated after training; calibration remains reserved.

Final test includes original-like central foci/fixed tissue values and the varied regime; circles and prespecified equal-nominal-area ellipses (axis ratio2.25); all SNR/radius/contrast conditions; three noise draws; independent healthy/perturbed noise as primary, common noise as sensitivity. This yields15,360 paired acquisitions ×14 methods (two conventional +12 checkpoints) =215,040 method-case rows, still only32 independent test groups. Shape/size can change sampled location and discrete area, so shape differences alone do not identify a pure shape mechanism.

All methods use identical measurements. Known simulated sigma is supplied only to Rician MLE. Conventional fits are cached only when exact magnitude bytes/shape, method and sigma match; this does not replace different observations. CNN evaluation is batched independently. Source/config/checkpoint hashes, source cleanliness, environment, seeds and registries preserve provenance.

## Null controls, aggregation and interpretation

δ=0 pairs have identical noiseless signals/references. Independent noise exposes spurious apparent changes; common-noise null changes are zero by construction for deterministic estimators. A prespecified benchmark null alarm is `|estimated focal mean change| > 1e-4 mm²/s`. It is not a clinical threshold or detection sensitivity. Record absolute null change and conditional finite-pair alarm rate, plus **alarm OR invalid focal pair** over every requested null pair. Invalid returns must not appear safer by disappearing from the denominator.

Report tissue/focal ADC bias/MAE/RMSE, finite maximum tissue error, signed recovery and absolute change error, opposite-sign recovery, nonfocal/spillover error, and invalid/undefined counts. Recovery is undefined for δ=0. Noise is averaged within each geometry/condition, then geometries equally; regime, shape, sign, null/change and pairing remain distinct. Compact summaries average conditions within each geometry first. Three optimization seeds describe sensitivity, not extra independent groups. Conditional errors/recovery use different surviving Rician estimates and require their denominators.

The richer training distribution and common composite selection differ from the first experiment. Historical before/after differences cannot isolate either change. The controlled effects are the four new cells on identical data/protocol. Spatial context, objective, optimization and low-SNR ambiguity may interact; better focal recovery may cost tissue precision or more null alarms. No calibrated confidence, adaptive stopping, acquisition reduction, anatomical-resource, clinical or publication-novelty claim follows.

## Frozen first-CNN development diagnosis

Verify the previous run's saved config/checkpoint hashes and unchanged essential model/simulation source. Use its original16 validation groups only, circles, both sizes/signs, zero noise and one SNR100 draw with independent paired noise. Averaging/log fitting must agree with the reference at zero noise. No calibration/test acquisitions enter diagnosis. High-SNR/noiseless inputs lie outside the original SNR5/20 training range; results are diagnostics, not a causal proof or a final performance estimate. These checks follow proposed [dMRI denoising evaluation criteria](https://pmc.ncbi.nlm.nih.gov/articles/PMC10402048/).

## Execute

Use the existing [learning-environment recipe and lock scope](FIRST_CNN_METHOD.md). Both commands create fresh output directories and preserve existing runs:

```sh
python -m adc_fidelity_bench diagnose-cnn --reference-run outputs/first-cnn --output outputs/first-cnn-development
python -m adc_fidelity_bench ablate-cnn --config configs/focal_ablation.json --output outputs/focal-ablation
```

The diagnostic command requires the local first-run checkpoints; these are intentionally not in public Git. The ablation is reproducible without them or any patient/external dataset. Preserve all checkpoints, raw CSVs/arrays and run manifests locally; only reviewed original numerical-summary figures accompany public documentation.
