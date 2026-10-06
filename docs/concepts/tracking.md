# Tracking and post-processing

!!! note "Planned"
    This explanation (with diagrams of gap filling and temporal padding) is
    written in step 4 from the implemented behaviour. Until then, the design is
    in [Pipeline § Post-processing](../design/pipeline.md#3-post-processing-no-gpu).

## Outline

1. Why per-frame detection is not enough
2. Tracks, observations and orphans
3. Gap filling
4. Temporal padding and spatial margins
5. 360° seam stitching
6. Sign deduplication
7. Known limitations
