# Training

SGBlur-Video **does not train its own detector** for now
([ADR-0009](../docs/adr/0009-model-registry-and-class-policy.md)): it uses the
YOLO26 model published by SGBlur, trained by the Panoramax team on their data.

This folder contains `MODEL_CARD.md`: what we know about the model in use.

To evaluate a registry model, use the benchmarks
([docs/guides/benchmarks.md](../docs/guides/benchmarks.md)):
`sgblur-video benchmark privacy --dataset … --model <name>` measures leakage on
the annotated privacy dataset, `sgblur-video benchmark speed` its speed.

Fine-tuning would only become necessary if the privacy benchmark shows
systematic misses that the pipeline cannot compensate (e.g. faces seen from
far away in 360° video). It would then require an NVIDIA GPU, datasets whose
licences allow redistributing weights (most face, plate and tracking datasets
are non-commercial — see `docs/research/step-1-analysis.md` §6), and would be
proposed upstream to SGBlur rather than maintained separately.
