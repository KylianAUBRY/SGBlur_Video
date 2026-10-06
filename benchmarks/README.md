# Benchmarks

How to build the annotated dataset and run the benchmarks:
[docs/guides/benchmarks.md](../docs/guides/benchmarks.md). Method and metrics:
[docs/design/testing-strategy.md](../docs/design/testing-strategy.md).

- `privacy-thresholds.yaml`: gates of the synthetic oracle (CI, every push) and
  of the privacy benchmark on the annotated dataset (`sgblur-video benchmark
  privacy`, run locally: the dataset contains personal data and never enters git).
- `results/`: reports of the benchmarks run for the project's decisions (numbers
  only, no picture, no file name).
