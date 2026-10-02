# Planned scientific tests

There are no scientific tests yet. Package/build checks do not validate ADC science. Before implementing each numerical/data component, write meaningful known-answer and adversarial tests for:

- signal generation, units, and zero-noise ADC recovery;
- nonpositive signals, degenerate b-values, fitting failures, and explicit exclusion policies;
- signal-level partial volume, including a mixture differing from blurred ADC;
- complex-to-magnitude noise, seed reproducibility, limiting/noise-floor behavior;
- tensor axes for b-values, repetitions, space, and later direction/coil;
- whole-anatomy/participant split isolation and train-only transforms;
- disjoint method-input/reference repetition IDs;
- focal-change signs, near-zero denominators, masks, and anatomy-level aggregation;
- coverage, false-confidence denominators, sequential stopping, and inclusion of non-stoppers in savings.

Add pytest and numerical dependencies with a reproducible environment at that milestone. Avoid tests that merely assert file existence or mirror constants.
