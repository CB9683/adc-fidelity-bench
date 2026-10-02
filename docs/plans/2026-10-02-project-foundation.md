# ADC Deep Learning Project Foundation Implementation Plan

> **SUPERSEDED, 2026-10-02:** The updated starter defines the scientific task and data strategy. This document is retained only as history; its unresolved-task sections must not guide implementation. Use [the corrected simulation-first plan](2026-10-02-trustworthy-adc.md).

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish a small, private-by-default research scaffold while the scientific question and dataset are defined.

**Architecture:** Use a Python package under `src/`, with separate configuration, tests, documentation, and generated outputs. Do not implement a scientific estimator, train a model, or publish until the relevant scientific and privacy decisions are recorded.

**Tech Stack:** Python, setuptools, JSON configuration, Markdown documentation, and Git when initialized. Numerical and learning dependencies remain undecided.

---

## Current evidence and boundaries

- Initial inventory: only `CODEX_STARTER_PROMPT.md`; no code, data, experiments, or Git repository.
- Intended owner: `CB9683`; proposed repository: `adc-deep-learning`. Final name and license require user confirmation before the first public push.
- No prediction target, cohort, acquisition protocol, reference standard, architecture, or performance claim is established.
- The user has authorized reading the starter and starting the project. The starter permits low-risk scaffolding after presenting this plan.
- Existing files must be preserved. The starter contains a local filesystem path and must remain outside published content.

## Task 1: Record the scientific specification and decisions

**Files:** Create `docs/PROJECT_CHARTER.md`, `docs/DECISIONS.md`, and `README.md`.

1. Record the objective as reproducible deep learning research for ADC analysis in diffusion MRI.
2. Describe the three candidate tasks without selecting one: estimate ADC from diffusion signals, predict outcomes using ADC, or improve ADC image quality.
3. Record pending decisions about scientific target, data access and restrictions, cohort boundaries, acquisition metadata, evaluation, repository name, license, and citation authorship.
4. Describe the implementation phases and explicitly identify unimplemented functionality.
5. Verify that no document asserts generated results, dataset access, an approved protocol, or an existing public repository.

## Task 2: Establish privacy and reproducibility boundaries

**Files:** Create `.gitignore`, `docs/DATA_LAYOUT.md`, and `configs/project.example.json`.

1. Ignore local prompts, real data folders, raw imaging formats, model weights, generated output, environment files, credentials, and editor/runtime state.
2. Require real participant data to live outside the repository and require permissions to use any proposed public dataset.
3. Define a provisional metadata checklist, including participant/session/scan grouping, image geometry, b-values, units, preprocessing, reference targets, and exclusions.
4. Provide a parseable configuration template with unknown scientific settings set to `null` and a reversible seed of `2026`.
5. Document participant-level split isolation, train-only preprocessing fitting, uncertainty reporting, and the pre-publication content/history audit.

## Task 3: Create an installable empty package

**Files:** Create `pyproject.toml`, `src/adc_deep_learning/__init__.py`, and `tests/README.md`.

1. Declare development version `0.1.0.dev0` and no numerical runtime dependencies.
2. Configure setuptools package discovery for `src/adc_deep_learning`.
3. Provide installation commands and identify the package as a scaffold, with no estimator, training command, or evaluation command yet.
4. Document future meaningful tests: known ADC values, invalid inputs and units, tensor shapes, participant split overlap, train-only fitted transforms, and evaluation aggregation.
5. Do not add tests that merely assert the existence of this scaffold.

## Task 4: Verify and review the scaffold

**Files:** Create `docs/FOUNDATION_CHECKS.md` after checks are run.

1. Parse configuration and packaging metadata with Python's standard library.
2. Build and install a wheel in temporary storage without installing into the user's global Python environment; import the installed package and verify its version.
3. Check `.gitignore` rules using a temporary Git repository and representative private filenames; do not stage or commit project files.
4. Review all new files for private data, secrets, machine-specific paths, false claims, and misleading license/release metadata.
5. Obtain an independent read-only review and address material findings.
6. Record exactly which checks ran and distinguish foundation verification from scientific validation.

## Subsequent phases: gated on scientific decisions

### Phase A: Freeze the initial study specification

Record the population, intended use, task, inputs, units, targets and their provenance, exclusions, dataset permissions, subject groups, split procedure, primary metric, uncertainty procedure, and intended contribution in `docs/PROJECT_CHARTER.md` and versioned configuration. Decide whether site, scanner, session, or temporal holdouts are also needed. Do not start training while these are unresolved.

### Phase B: Implement the conventional baseline and data handling

For ADC estimation, evaluate an agreed monoexponential conventional fit before introducing deep learning. For outcome prediction, define an appropriate conventional feature/model comparator. For image enhancement, include the unmodified input and an agreed conventional method. Use @superpowers:test-driven-development for numerical and data logic, beginning with synthetic known-answer inputs and leakage tests. Record exact preprocessing, b-values, units, invalid-signal handling, and any fitting constraints. File names and algorithm details will be planned after the task is confirmed.

### Phase C: Add the smallest justified deep-learning experiment

Choose the architecture only after the input/target contract and baseline are verified. Version configuration, split manifests, software versions, seeds, checkpoints, and run metadata outside published participant data. Keep validation-based choices separate from the held-out test evaluation.

### Phase D: Evaluate and characterize limitations

Report task-appropriate performance, participant-level uncertainty, diagnostic plots, baseline comparisons, and sensitivity to noise, b-values, preprocessing, scanner/site differences, and exclusions where supported by the dataset. Synthetic recovery establishes implementation behavior, not clinical or biological validity.

### Phase E: Prepare public release

Confirm repository name, license, authorship, and permissions for every included asset. Add the approved license, finalized `CITATION.cff`, locked scientific environment, verified commands, and appropriate CI. Inspect staged content and complete Git history for secrets, identifiable or restricted information, large files, and local paths; show the audit result before the first public push. Create the public repository only within the user's confirmed release choices.
