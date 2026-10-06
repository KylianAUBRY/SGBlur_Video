# Training

SGBlur-Video **does not train its own detector** for now
([ADR-0009](../docs/adr/0009-model-registry-and-class-policy.md)): it uses the
YOLO26 model published by SGBlur, trained by the Panoramax team on their data.

This folder will contain:

- `MODEL_CARD.md` — what we know about the model in use (see below);
- `evaluate.py` (step 8) — evaluation of a registry model on the annotated privacy dataset.

Fine-tuning would only become necessary if the privacy benchmark shows
systematic misses that the pipeline cannot compensate (e.g. faces seen from
far away in 360° video). It would then require an NVIDIA GPU, datasets whose
licences allow redistributing weights (most face, plate and tracking datasets
are non-commercial — see `docs/research/step-1-analysis.md` §6), and would be
proposed upstream to SGBlur rather than maintained separately.
