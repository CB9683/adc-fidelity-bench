# Foundation verification record

Date: 2026-10-02. Scope: historical initial scaffold and proposed protocol, not scientific validation. The subsequent executable CPU implementation and numerical evidence are recorded in [pilot results](PILOT_RESULTS.md); statements below describe their original verification stage.

The outcomes below record the original foundation checks, performed with the `adc-deep-learning` distribution and `adc_deep_learning` import before a code license was selected. They are historical results, not verification of the subsequent rename. The user has since confirmed `CB9683/adc-fidelity-bench` and the MIT code license; the current import is `adc_fidelity_bench`. The renamed package and new license/citation files require their own verification before release.

## Verified outcomes

| Check | Result |
| --- | --- |
| JSON/TOML parsing and package Python syntax | Passed |
| Explicit draft status, external paths, generator/preprocessing distinction | Passed |
| Package wheel build without network or global installation | Passed |
| Wheel installation/import in an isolated environment | Passed; version `0.1.0.dev0` |
| Wheel contents | Package present; no starter or imaging/array data files |
| Local Markdown links | Passed |
| Final ignore-rule probes | 24 representative private paths ignored; 7 public paths allowed |
| Public-candidate text inspection before this record | 14 files; no matching common secret/local-path patterns or files over 1 MB |
| Independent read-only review | Safe-start scope met; two clarifications addressed |

Build used local Python 3.12.2, pip 24.0, and setuptools 80.9.0. Installation/import used a temporary Python 3.13.7 virtual environment. Git ignore checks used Git 2.50.1. Python 3.11 compatibility is declared but has not been exercised here. No scientific dependencies were installed.

## Commands and procedures

JSON was parsed with `json.loads`, TOML with `tomllib.loads`, and the package with `ast.parse`. Protocol checks verified `abs(delta_ref) > epsilon`, the superseded-plan marker, a fixed generator reference amplitude, and unresolved estimator normalization.

For packaging, copy `pyproject.toml`, `README.md`, and `src/` into a temporary source directory. The executed wheel command was equivalent to:

```sh
python3.12 -m pip wheel --no-deps --no-build-isolation --no-index \
  --wheel-dir TEMP/wheels TEMP/source
```

`TEMP` denotes temporary storage, not a literal working path. The build backend was already available locally. Inspect the resulting wheel with `zipfile`, then install that wheel using `--no-deps --no-index` into a newly created temporary Python 3.13 virtual environment. Import from outside the source tree and verify both package and distribution versions equal `0.1.0.dev0`. Temporary files/environments were removed afterward.

Ignore rules were copied into a temporary Git repository created with `git init -q`. For each probe use:

```sh
git -C TEMP -c core.excludesFile=/dev/null check-ignore --no-index -q PATH
```

Expected exit code is 0 for ignored private paths and 1 for public source/docs/config paths. Probes covered the local starter, dataset/anatomy/simulation folders, private manifests, environment files, local configuration, credentials, imaging/array files, model weights, and outputs. Public README, package metadata/source, test/config/doc paths, and citation template remained allowed. No project files were staged or committed.

Inspect public-candidate text for common home/mount-path patterns, cloud/API token prefixes, private-key headers, and file sizes; exclude the local ignored starter from public candidates. Check relative Markdown links against the filesystem. This pattern check is heuristic and does not establish the absence of all private information.

## Review corrections

- Recovery-ratio eligibility now uses absolute reference change, covering both positive and negative focal ADC changes.
- Fixed simulation amplitude is separate from observed-input estimator normalization; no per-voxel truth-based normalization is authorized.

## Limits and next gate

### Rename and MIT verification, 2026-10-02

The approved project is now `adc-fidelity-bench`, imported as `adc_fidelity_bench`. The renamed wheel built successfully and installed/imported in a temporary Python 3.13.7 environment. Wheel metadata reports `License-Expression: MIT` and `License-File: LICENSE`; the packaged license matches the repository's MIT license exactly. The old import package is absent. `CITATION.cff` parsed successfully as YAML and uses the public GitHub handle `CB9683` without invented personal names or a DOI.

This build used Python 3.12.2, setuptools 80.9.0, and packaging 26.0 from a temporary overlay. An initial offline build exposed packaging 24.0 in the local Python 3.12 environment, which lacks the SPDX license support required by setuptools 77+. Reusing compatible local packaging 26.0 in temporary storage resolved the issue. No global packages were changed. Normal installation uses the default isolated build environment; `--no-build-isolation` requires compatible build tools to be supplied explicitly.

Git initialization and eventual publication are subsequent steps, separate from the original foundation checks above. The name and MIT license have been confirmed by the user; research functionality remains unimplemented.

No simulation, ADC estimator, numerical tests, training, calibrated intervals, stopping experiments, or scientific results exist yet. Package checks do not validate these methods. At the original foundation check, no datasets had been downloaded, no access approval obtained, and no repository published or Git history created. Repository setup and the required staged-content and complete-history audit follow these recorded foundation checks.

Next review the benchmark specification, physical/tissue/focal parameter ranges, stopping-error quantity, exploratory tolerance/risk settings, and sequential calibration strategy. Then prepare the detailed test-first simulation/baseline implementation plan and locked numerical environment. The MIT code license is confirmed; initial citation metadata uses the owner handle `CB9683`, with fuller authorship and release metadata still pending.
