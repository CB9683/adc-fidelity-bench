# Decision register

Updated 2026-10-02 from the revised brief and the user's repository-name and license approval. Previous generic task-category/data questions are superseded.

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
| Tissue/focal ranges and spatial geometry | Declared numerical assumptions | Original two-region ellipses with synthetic ADC/S0; circles and signed focal contrast; biological ranges remain unresolved |
| Baselines | Two implemented | Magnitude average + log fit; known-sigma repetition-level Rician MLE; published denoising next |
| Architecture | Deferred | Smallest justified method after baseline/evaluation contracts |
| Stopping error, tolerance, and risk | Pending review | Supplied operating point or declared exploratory sweeps |
| Sequential calibration | Pending implementation | Evaluate complete stopping policy on independent cases |
| Repository/license | Confirmed and public | `CB9683/adc-fidelity-bench`; MIT code license |
| Citation | Initial metadata | Owner identified by GitHub handle `CB9683`; fuller authorship and release metadata pending |
| Novelty | Not established | Focused overlap review documented; broader check needed |

Any existing prespecified stopping tolerance/risk remains useful user input. Without a supplied tolerance, use a clearly declared exploratory sweep. Resource permissions and real-data approval must precede use. Large-scale training awaits quantitative protocol review. The small CPU pilot follows numerical/specification review and a public-content audit; it makes no calibrated-stopping or clinical claim.
