# Licence

!!! info "Not legal advice"
    This page explains the licensing situation as understood by the
    maintainers. Seek legal advice for your own deployment.

## This repository

The source code of SGBlur-Video is released under the
[MIT licence](https://github.com/KylianAUBRY/SGBlur_Video/blob/main/LICENSE)
(decision: [ADR-0006](adr/0006-licence.md)), like SGBlur.

## Runtime dependency on Ultralytics (AGPL-3.0)

Detection and tracking use [Ultralytics](https://github.com/ultralytics/ultralytics),
licensed under the **GNU AGPL-3.0** (or an Ultralytics Enterprise licence).
MIT code can be combined with AGPL-3.0 code, but a running service that imports
Ultralytics forms a combined work subject to the AGPL-3.0 conditions of that
dependency. In practice, whoever deploys SGBlur-Video as a network service
should offer its users the corresponding source code of what runs (this
repository, Ultralytics, and any modification), or hold an Ultralytics
Enterprise licence.

Ultralytics' [licence page](https://www.ultralytics.com/license) states that
projects using their code or trained models must be AGPL-3.0 as a whole or
licensed commercially. That is the vendor's reading; this project does not
settle the question.

## Model weights

SGBlur-Video does not distribute model weights. They are downloaded from the
SGBlur repository at a pinned commit. Upstream statements differ: the
Hugging Face card of `Panoramax/detect_face_plate_sign` says *etalab-2.0*,
while the checkpoint metadata written by Ultralytics says *AGPL-3.0*. Both are
reproduced in
[THIRD_PARTY_LICENSES.md](https://github.com/KylianAUBRY/SGBlur_Video/blob/main/THIRD_PARTY_LICENSES.md).

## Other components

PyAV (BSD-3-Clause) bundles FFmpeg libraries built with `libx264`/`libx265`
(GPL); PyTorch (BSD-style), OpenCV (Apache-2.0), FastAPI (MIT), Pydantic (MIT).
Docker images, which bundle all of them, are distributed under the terms of
their components. The full list is in `THIRD_PARTY_LICENSES.md`.
