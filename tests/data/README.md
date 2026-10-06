# Test data

No video or picture is committed to this repository: videos can show
identifiable people, and they are large.

* Small free-licence fixtures (`gopro/gpmf-parser` samples) are downloaded by
  `tests/data/fetch.py` from pinned URLs, verified by SHA-256 and cached in
  `~/.cache/sgblur-video/fixtures`. Their licences are listed in `THIRD_PARTY_LICENSES.md`.
* Synthetic videos for the privacy oracle (flat and 360°) are generated on the fly by
  `tests/privacy/synthetic.py`.
* The manually annotated privacy dataset lives **outside** the repository; see
  `docs/design/testing-strategy.md`.
