---
status: accepted (implemented in steps 4–6)
date: 2026-10-06
---

# Irreversible blur method and shapes

> **Update 2026-10-08.** Since [ADR-0012](0012-independent-frames.md), every blurred region is a rectangle exactly on the detected box (no ellipse, no margin); the blur methods are unchanged.

## Context and Problem Statement

A light Gaussian blur can be partially inverted (deconvolution) or defeated by
recognition models trained on blurred faces. What operation and shape do we
apply to faces and plates?

## Decision Drivers

- Irreversibility (EDPB Guidelines 3/2019: blurring without retroactive recovery counts as erasure).
- Coverage: the shape must contain the whole detected box plus margin.
- Visual acceptability for Panoramax users.
- Speed at 8K, 10-bit, without colour shifts.

## Considered Options

1. Gaussian blur only.
2. Pixelation (mosaic) with few cells, then Gaussian blur (SGBlur does pixelate + box blur).
3. Solid fill.
4. Inpainting / synthetic face replacement.

## Decision Outcome

Chosen option: **2 by default** (`BLUR_METHOD=pixelate_blur`: the region is
averaged down to at most `PIXELATE_CELLS=6` cells on its long side, smoothed at
that low resolution and interpolated back bilinearly), with **3** (`solid`) and
a strong Gaussian variant (`gaussian_strong`, also computed at reduced
resolution) selectable.

Update (step 4): the first implementation applied a full-resolution Gaussian
(σ = cell/2) after the mosaic. On 8K frames it cost ~70 ms per region (3 s per
frame); smoothing the ≤ 6×6 averages before upscaling gives the same visual
result (no block edges) and the same information bound in microseconds.

Shapes: faces use the **ellipse circumscribing** the margin-enlarged box
(semi-axes √2 × half-sides — an inscribed ellipse would leave the 21.5 %
corner area unblurred); plates use the rectangle. Operations run on the YUV
planes in the source bit depth (luma full resolution, chroma subsampled mask),
never on decoder-owned buffers.

### Consequences

- Good: mosaic with ≤ 6 cells keeps very little identity information; blur removes block edges that help super-resolution models.
- Good: no RGB round trip — no colour shift, half the memory traffic at 8K 10-bit.
- Bad: mosaics with more cells have been partly re-identified by ML (McPherson, Shokri, Shmatikov 2016, arXiv:1609.00408); kept low by default and documented.
- Bad: averaging many frames of the same blurred face may leak information (multi-frame attacks); `solid` removes that risk and is documented as the maximum-guarantee option.
- Option 4 rejected: needs another model, slower, and a failure is worse than a blur.
