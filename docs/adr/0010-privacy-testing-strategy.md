---
status: accepted
date: 2026-10-06
decision-makers: maintainer (Kylian Aubry)
---

# Privacy testing: synthetic oracle + annotated real clips

## Context and Problem Statement

The brief requires a privacy non-regression test: the share of frames where a
ground-truth face or plate is not covered by the blur, with CI failing above a
threshold. Ground truth cannot come from our own model. Which references do we
use?

## Considered Options

- **A** Synthetic oracle: generated videos with known object positions and a scripted fake detector.
- **B** Independent second detector (EgoBlur, Apache-2.0) as a reference.
- **C** Visual review of debug videos.
- **D** Manually annotated real clips.

## Decision Outcome

Chosen: **A + D** (maintainer decision). Details in
[testing-strategy.md](../design/testing-strategy.md).

- **A** runs in public CI on every push, needs no GPU and no personal data, and checks the whole chain down to the encoded file. Gate: zero leaked object-frames for objects reported at least once by the fake detector, zero blurred sign pixels outside face/plate overlaps.
- **D** uses 3–5 short clips from the maintainer's own videos, annotated in CVAT (track mode with interpolation), stored **outside git**. Metrics: coverage-based protection (≥ 0.9 of the box), leakage rate by class/size/readability, tracks ever leaked, longest exposure, transient exposures, over-blur ratio. Run on demand and before any change of defaults.

### Consequences

- Good: A catches regressions of tracking/post-processing/rendering immediately; D measures the real detector.
- Bad: D cannot run on public CI (personal data); it runs locally or on a self-hosted runner. Its size limits statistical power at first.
- Bad: pre-annotation by our model can bias annotators; mitigated by explicit instructions to look for misses and a second reviewer when possible.
- B and C remain available as tools (`--debug` videos are produced by the CLI anyway).
