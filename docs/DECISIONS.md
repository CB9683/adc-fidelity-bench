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
| Initial noise/partial volume | Proposed | Complex noise before magnitude; signal mixing/resampling |
| Pilot settings | Reversible | Example configuration; not clinical acceptance criteria |
| Tissue/focal ranges and spatial geometry | Pending protocol specification | Literature-informed assumptions before generation |
| Baselines | Defined categories | Averaging + fitting, noise-aware fitting, published denoising |
| Architecture | Deferred | Smallest justified method after baseline/evaluation contracts |
| Stopping error, tolerance, and risk | Pending review | Supplied operating point or declared exploratory sweeps |
| Sequential calibration | Pending implementation | Evaluate complete stopping policy on independent cases |
| Repository/license | Confirmed by user | `CB9683/adc-fidelity-bench`; MIT code license; initial scaffold prepared for publication |
| Citation | Initial metadata | Owner identified by GitHub handle `CB9683`; fuller authorship and release metadata pending |
| Novelty | Not established | Focused overlap review documented; broader check needed |

Any existing prespecified stopping tolerance/risk remains useful user input. Without a supplied tolerance, propose an exploratory sweep for review. Resource permissions and real-data approval must precede use. Large-scale training and scientific-result releases await their stated review gates. The initial source/documentation scaffold can be published after the privacy audit under the confirmed name and MIT license.
