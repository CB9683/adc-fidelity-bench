# First small CNN: frozen numerical experiment

The first learned estimator directly predicts acquisition-resolution ADC. It uses the same original geometric simulator and reference definition as the conventional pilot. This is a fixed-budget learning experiment; uncertainty calibration and stopping are separate, unimplemented work.

## Measurement and supervised target

Each input contains four observed magnitude repetitions at each of b=0 and 1000 s/mm²: eight acquisitions in total. The two channel images are the magnitude averages at each b-value. Divide both by the 95th percentile of the observed mean-b0 image, floored at 1e-6. This statistic is calculated anew from that input only. No noiseless amplitude, true ADC, tissue/focal mask, generator sigma, paired healthy reference or additional repetitions enters inference.

The target is ADC fitted to noiseless diffusion signals after 2×2 spatial signal averaging. It is not blurred assigned ADC. Fine numerical geometries are 16×16 at 0.5 mm; acquisition maps are 8×8 at 1 mm. Synthetic S0/ADC and the single-coil complex-noise approximation match the pilot. Tissue footprints with positive fine-grid tissue fraction define training loss support. References outside tissue are NaN and are excluded before subtraction.

## Network and optimization

The network is Conv(2→16,3×3,padding1), ReLU, Conv(16→16,3×3,padding1), ReLU, Conv(16→1,1×1), Softplus×1e-3 mm²/s. It has 2,641 trainable parameters and 5×5 spatial context. There is no pooling, perceptual loss or explicit smoothness penalty.

Loss is tissue MSE of ADC scaled by 1e3, calculated per image before averaging images equally. Both healthy and perturbed cases contribute. Adam uses lr 0.001, batch 64, 80 epochs maximum. Three seeds 11/22/33 run independently; each checkpoint is selected by lowest validation loss, with the earliest epoch retained on an exact tie. Save every seed's history and reload the selected checkpoint using PyTorch's restricted `weights_only=True` loader before evaluation. Final-test outcomes never select seeds, epochs, architecture or loss.

The frozen CPU execution uses two threads and deterministic PyTorch operations. Seeds and settings improve within-environment repeatability; they do not promise bit-identical behavior across releases/hardware, consistent with [PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html). GPU support is a setup capability, not a reason to prefer GPU for this tiny run.

## Independent groups and held-out tests

The new cohort has 144 IDs (`cnn-000`…`cnn-143`), with seed 20261002 and 64 training, 16 validation, 32 reserved calibration and 32 final-test geometry groups. Split before generating derivatives; no old pilot test IDs are reused. All focal/noise variants of an anatomy stay in its role. Training uses four noise draws per condition and validation two; final evaluation uses one. Calibration groups are not generated or inspected.

Training/validation use circles with radius 0.5/1.5 mm, ADC change±0.0003 mm²/s and reference SNR 5/20. Ellipses are registered before final evaluation and are absent from training/validation. Their nominal area matches the circle, with major/minor axis ratio 2.25. Both common and independent healthy/perturbed noise pairing are tested. Shape and pairing registries are explicit in the saved configuration.

There are 4,096 training images and 512 validation images. Final evaluation has 1,024 paired acquisitions (32 groups × 2 shapes × 2 pairings × 2 SNRs × 2 radii × 2 contrasts), each scored by averaging/log fitting, known-sigma Rician MLE and each of the three CNN checkpoints: 5,120 method-case rows. Those rows still represent only 32 independent test geometry groups; the three optimization seeds do not create additional anatomies.

## Quantitative comparison and limitations

All methods use identical observed acquisitions. The Rician estimator alone receives known simulated sigma; the CNN and log fit do not. State this information advantage rather than attributing every difference to architecture.

Report ADC bias/MAE/RMSE, focal and nonfocal ADC errors, signed paired focal recovery/change error, nonfocal paired error and invalid/focal-failure counts. An incomplete paired focal ROI makes full-region recovery undefined. Average noise within anatomy/condition, then anatomies equally; compact summaries first average conditions within each anatomy. Preserve defined-case/anatomy denominators. Neither lower conditional RMSE nor a finite CNN map alone establishes quantitative trustworthiness.

The fixed central focal design and limited two-tissue geometric family allow strong spatial priors. Shape holdout remains within that family; it is not a realistic anatomical or acquisition shift. MSE can favour prior-driven estimates that suppress a small real change. The experiment is designed to expose that possibility. A finite positive CNN output is not an identification diagnostic, an uncertainty bound or evidence that fewer acquisitions are safe. No patient/anatomical-resource, clinical or prospective scanner-time claim follows.

Published-denoiser comparison, more varied anatomy/lesion locations and acquisition regimes, and independent uncertainty calibration/stopping remain later work. This first CNN is a transparent research-software milestone, not a settled publication contribution.

## Reproduce

On macOS ARM64 with Python 3.13, install `requirements-learning-lock.txt` and the package. This exact 27-package closure is scoped to that environment. Baseline users can retain `requirements-lock.txt`; PyTorch is optional. Linux CI installs a CPU PyTorch wheel from the [official CPU index](https://download.pytorch.org/whl/cpu/torch/) and checks training separately.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-learning-lock.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m pytest
.venv/bin/python -m adc_fidelity_bench train-cnn --config configs/first_cnn.json --output outputs/first-cnn
```

Choose a fresh output directory for each run. Outputs preserve exact configuration, independent split registry, source/config/checkpoint hashes, environment/device, training examples and losses, selected tensor/state checkpoints, quantitative CSVs and diagnostic figures. Checkpoints and generated arrays remain outside Git.
