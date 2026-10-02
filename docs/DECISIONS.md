# Decision register

Updated 2026-10-02 from the revised brief, the user's repository-name and license approval, the first trained CNN and the frozen four-cell follow-up. Previous generic task-category/data questions are superseded.

| Decision | Status | Current position |
| --- | --- | --- |
| Task | Defined | Trustworthy ADC estimation with fewer repeated acquisitions |
| Main failure | Defined | Focal-change suppression with unjustified confidence |
| Initial evidence | Defined | Controlled simulations; no patient dependency |
| Later real data | Access pending | fastMRI Prostate; disjoint input/reference repetitions |
| Signal model | Defined initial choice | Monoexponential tissue signals |
| Anatomy | Proposed; terms unresolved | BrainWeb structure with synthetic diffusion parameters |
| Initial noise/partial volume | Implemented and reviewed | Independent complex Gaussian components after signal-level 2×2 block averaging; magnitude formation afterward |
| Pilot settings | Executable, exploratory | `configs/pilot.json`; no clinical acceptance criteria |
| Tissue/focal ranges and spatial geometry | Declared numerical assumptions | Original two-region geometric anatomy; the follow-up varies tissue ADC/S0 and valid focal locations, includes both contrast signs and zero-change controls, and holds focal ellipses out of training; biological ranges remain unresolved |
| Baselines | Two implemented | Magnitude average + log fit; known-sigma repetition-level Rician MLE; published denoising next |
| Architecture | First CNN trained and evaluated | 2,641-parameter direct ADC estimator; three validation-selected seeds; lower tissue error but strong focal attenuation; see [results](FIRST_CNN_RESULTS.md) |
| Four-cell comparison | Implemented under frozen protocol | Spatial 2,641-parameter versus pointwise 2,647-parameter models × tissue MSE versus tissue-plus-paired-change MSE; three seeds per cell; see [method](FOCAL_PRESERVATION_METHOD.md) and [results](FOCAL_PRESERVATION_RESULTS.md) |
| Follow-up model selection | Frozen and shared across cells | Earliest minimum validation tissue MSE + 1.0×paired-change MSE; all 12 checkpoints reported; no final-test selection |
| Follow-up groups and pairing | Fresh grouped design | 64/16/32/32 new train/validation/reserved-calibration/test groups; calibration acquisitions unused; independent paired noise primary, common noise sensitivity |
| Zero-change control | Prespecified benchmark endpoint | Alarm at absolute estimated focal mean change >1e-4 mm²/s; report conditional finite-pair alarms and alarm OR invalid over all requested null pairs; no clinical or calibrated-risk interpretation |
| Stopping error, tolerance, and risk | Pending review | Supplied operating point or declared exploratory sweeps |
| Sequential calibration | Pending implementation | Evaluate complete stopping policy on independent cases |
| Repository/license | Confirmed and public | `CB9683/adc-fidelity-bench`; MIT code license |
| Citation | Initial metadata | Owner identified by GitHub handle `CB9683`; fuller authorship and release metadata pending |
| Novelty | Not established | Focused overlap review documented; broader check needed |

Any existing prespecified stopping tolerance/risk remains useful user input. Without a supplied tolerance, use a clearly declared exploratory sweep. Resource permissions and real-data approval must precede use. Large-scale training awaits quantitative protocol review. The small CPU conventional pilot, first CNN and controlled four-cell study follow numerical/specification review and public-content audits; none makes a calibrated-stopping or clinical claim. Historical before/after comparisons do not isolate richer-data effects; controlled architecture/objective comparisons use the same new cohort.
