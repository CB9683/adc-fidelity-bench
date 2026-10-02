# Data, privacy, and provenance

## Public content and release review

Public content is code, configuration templates, tests, and approved documentation. Keep patient data, identity mappings, private metadata, restricted manuscripts, credentials, anatomy downloads, and local filesystem paths outside Git. The local starter is ignored because it contains a machine-specific path.

`.gitignore` is a guardrail, not de-identification or proof of publication safety. Never force-add private files. Before the first public push, inspect staged content and every commit in the complete history, including notebook outputs and metadata, for sensitive material, secrets, large files, and local paths. Show the result and confirm name, license, and asset permissions.

## External working storage

Proposed relative layout outside the repository:

```text
anatomy/       Resource files, checksums, and use terms
simulations/   Signals, reference maps, masks, and generation metadata
manifests/     Private split and sample manifests
real/          Approved fastMRI files and preprocessing, later
runs/          Config snapshots, versions, metrics, plots, checkpoints
```

Paths belong in ignored `configs/*.local.json` files or external configuration. Public templates leave them `null`. Review synthetic/anatomically derived outputs for source-resource rights before publication; public availability does not imply redistribution permission.

## Provenance required before experiments

Simulation records: anatomy/source/version/checksum/use terms; seed/generator version; assigned `S0` and ADC with units; b-values; physical geometry; tissue fractions; signal-resampling operator; focal perturbation; SNR and complex-noise definitions; coil/reconstruction assumptions; repetition IDs; reference fit; exclusions. Preserve both assigned parameters and signal-derived acquisition-resolution reference.

Real records: pseudonymous participant, session and scan IDs; direction/b-value/repetition ID; image geometry; ADC scaling/units; reconstruction/coil information; preprocessing; quality exclusions; access restrictions. Identity links stay in approved private storage. Participant manifests must not enter Git.

## Split and repetition integrity

Group all simulation variants by source anatomy before generation. Use disjoint train, validation, calibration, and test anatomy sets, with separate unfamiliar-regime evaluation.

Group all real sessions/scans by participant. Reserve disjoint input/reference repetition IDs per b-value/direction. Budget curves must respect those reservations; an all-repeat map overlaps any subset used as input and cannot be an independent reference.

No datasets have been downloaded or access applications submitted. Verify current terms before use. DIPY or reconstruction-code licenses do not confer dataset redistribution rights.
