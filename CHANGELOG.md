# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Step 1 analysis of SGBlur, the Panoramax blur API and Ultralytics tracking (`docs/research/`).
- Design documents (`docs/design/`): architecture, pipeline, `detections.jsonl` v1 format, HTTP API contract (OpenAPI 3.1 draft), configuration, testing strategy.
- Architecture decision records ADR-0001 to ADR-0010 (`docs/adr/`).
- Project skeleton: `pyproject.toml` (Python 3.14, uv), package layout, typed configuration with generated reference, model registry with SGBlur `yolo26s` weights (pinned URL and SHA-256), recall-oriented tracker configurations, CLI skeleton, unit tests.
- Tooling: ruff, mypy (strict), pytest with coverage, pre-commit, GitHub Actions CI, MkDocs Material documentation site.
- Community files: README (English and French), contributing guide, code of conduct, security and privacy policy (public issue template without media in v1; private reporting planned for v2), third-party licences.
