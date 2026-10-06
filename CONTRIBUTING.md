# Contributing to SGBlur-Video

Thank you for helping! SGBlur-Video protects the privacy of people filmed in
street-level videos: **a change that may leave a face or a plate visible is a
bug, even if it makes everything else better.** Keep that in mind when you
touch detection thresholds, tracking or blurring.

## Ground rules

- Be kind: this project follows the [code of conduct](CODE_OF_CONDUCT.md).
- **Never commit or attach real videos or pictures of people** — not in the repository, not in issues, not in pull requests, not in CI logs. Use synthetic data (see `tests/`) or describe the situation in words.
- **Privacy leaks** (something left unblurred) are reported with the dedicated issue template, **without any media or identifying detail**: see [SECURITY.md](SECURITY.md) (private reporting is planned for v2).
- Discuss large changes in an issue first; structural decisions get an ADR in `docs/adr/`.

## Development setup

Requirements: Python 3.14, [uv](https://docs.astral.sh/uv/), git.

```bash
git clone https://github.com/KylianAUBRY/SGBlur_Video.git
cd SGBlur_Video
uv sync                         # creates .venv with dev and docs tools
uv run pre-commit install       # lint, format, types and docs checks on commit
uv run pytest -m "not integration"   # fast, offline: unit tests and the synthetic privacy oracle
uv run pytest                   # everything (about 30 s once the downloads are cached)
```

Test folders and markers:

| Folder | Marker | Content |
|---|---|---|
| `tests/unit` | — | Pure logic: geometry, post-processing, formats, store, CVAT, metrics… |
| `tests/privacy` | `privacy` | Synthetic privacy oracle and benchmark: the CI privacy gate. No network. |
| `tests/integration` | `integration` (and `slow`) | Real model, real media and the HTTP API. Downloads the model (~20 MB, GitLab) and a free GoPro sample (GitHub) once, into `~/.cache/sgblur-video`. |

Each test is stopped after 300 s (`pytest-timeout`), and warnings are errors.

On Linux, the lock file installs CPU-only PyTorch wheels; NVIDIA GPUs are used
through the `gpu` Docker image ([installation](docs/getting-started/installation.md#nvidia-gpu)).
On macOS, if `import sgblur_video` fails right after the first `uv sync`, see
[troubleshooting](docs/guides/troubleshooting.md).

## Before opening a pull request

```bash
uv run ruff format .
uv run ruff check .
uv run mypy                      # strict, on src/
uv run pytest --cov
uv run mkdocs build --strict     # documentation must build without warnings
```

The CI runs the same commands. If you change `src/sgblur_video/config.py`,
regenerate the configuration reference:

```bash
uv run python scripts/gen_config_reference.py
```

## Conventions

- **Language**: code, docstrings, comments, commit messages and docs in English; `README.fr.md` is the French entry point.
- **Docstrings**: Google style on every module, class and public function (args, returns, raises, a short example when useful). Comments explain *why*, not *what*.
- **Types**: annotations everywhere; `mypy --strict` must pass on `src/`.
- **Classes are referenced by name** (`"face"`, `"plate"`…), never by model index.
- **No hard-coded model names**: models come from `models/registry.yaml`.
- **Ultralytics** is pinned exactly; it is only used through the adapters in `sgblur_video.core` so that upgrades are tested in one place.
- **Logs** never contain file names, upload paths, coordinates or images.
- **Tests**: unit tests for every behaviour; post-processing and blur logic must be testable without GPU and without real videos. Coverage ≥ 85 % on `core/`, `privacy/` and `semantics/`.
- **Commits**: small and focused, with a [Conventional Commits](https://www.conventionalcommits.org/) prefix and an imperative subject (`feat: link sign fragments across the 360° seam`, `fix: …`, `docs: …`).
- **Changelog**: add a line under `Unreleased` in [CHANGELOG.md](CHANGELOG.md) for user-visible changes.

## Pull request process

1. Fork, create a branch from `main`, make your change with tests and docs.
2. Open a pull request describing the change and its privacy impact (fill in the template).
3. A maintainer reviews it. Changes to defaults that affect privacy need the privacy benchmark results (see `docs/design/testing-strategy.md`).
4. Once approved and green, a maintainer merges it.

## Project structure

See [docs/design/architecture.md](docs/design/architecture.md#package-layout)
for the package layout and [docs/adr/](docs/adr/README.md) for the decisions
behind it.
