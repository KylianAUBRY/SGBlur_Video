# Test data

No video or picture is committed to this repository: videos can show
identifiable people, and they are large.

* Small free-licence fixtures (e.g. `gopro/gpmf-parser` samples, CC0 clips) will
  be downloaded by `tests/data/fetch.py` from pinned URLs, verified by SHA-256 and
  cached (step 4). Their licences are listed in `THIRD_PARTY_LICENSES.md`.
* Synthetic videos for the privacy oracle are generated on the fly by
  `tests/privacy/synthetic.py` (step 4).
* The manually annotated privacy dataset lives **outside** the repository; see
  `docs/design/testing-strategy.md`.
