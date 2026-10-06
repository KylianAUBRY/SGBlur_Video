---
status: accepted
date: 2026-10-06
decision-makers: maintainer (Kylian Aubry)
---

# Licence: MIT code with an AGPL-3.0 runtime dependency

## Context and Problem Statement

The project brief proposed AGPL-3.0-or-later because Ultralytics (library and,
according to Ultralytics, trained weights) is AGPL-3.0. The repository already
carries an MIT `LICENSE`, like SGBlur, which also depends on Ultralytics.

## Decision Outcome

The maintainer chose to **keep the MIT licence** for this project's code
(2026-10-06), the same situation as SGBlur.

### Consequences (to document in `docs/license.md` in step 3)

- This repository's source code is MIT. MIT code can be combined with AGPL-3.0 code.
- A running service that imports `ultralytics` forms a combined work subject to the AGPL-3.0 conditions of that dependency: whoever deploys it as a network service must offer users the corresponding source of the combined program (this repository + Ultralytics + any modifications), or hold an Ultralytics Enterprise licence. Ultralytics' own licence page states that projects using their code or models must be AGPL-3.0 as a whole or licensed commercially; that is their reading, and this record does not settle it (not legal advice).
- Model weights are not distributed by this project: they are downloaded from SGBlur's repository at a pinned commit. Their licensing is upstream's statement (Hugging Face card: etalab-2.0; checkpoint metadata: AGPL-3.0) and is reproduced as-is in `THIRD_PARTY_LICENSES.md`.
- Other runtime components: PyAV (BSD-3-Clause) with bundled FFmpeg built with libx264/libx265 (GPL in practice), OpenCV (Apache-2.0), PyTorch (BSD-style), FastAPI (MIT). Docker images, which bundle all of these, are distributed under the terms of their components; the image documentation lists them.
- Test data and fixtures are listed with their licences in `THIRD_PARTY_LICENSES.md`.
