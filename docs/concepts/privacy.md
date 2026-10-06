# Privacy and GDPR

Street-level videos film people who did not consent. SGBlur-Video exists to
protect them: **a face or a plate left visible on one frame is a failure, even
if every other frame is perfect.** This page explains what the service does
with personal data and where its limits are. It is not legal advice: the
operator of an instance remains responsible for its GDPR compliance.

## Recall over precision

Every design choice favours blurring too much over blurring too little
([Tracking and post-processing](tracking.md)):

- detection runs on every frame, at several scales, with tiles on 8K footage;
- low-score detections are blurred too (`CONF_BLUR`, default 0.15), including
  isolated ones that no tracker followed;
- fragments of the same object are linked over gaps of up to 1 s, gaps are
  filled by interpolation, and every chain is blurred 15 frames before its
  first and after its last detection, with a growing box;
- boxes get a 15 % margin and faces the ellipse circumscribing that box;
- a model that cannot detect a class configured to be blurred is refused at
  start-up ([ADR-0009](../adr/0009-model-registry-and-class-policy.md)).

The cost is over-blurring: shop posters, statues or road markings mistaken for
faces or plates are blurred as well. Traffic signs are never blurred.

## Irreversible blur

Regions are pixelated to at most 6 cells across, smoothed and upscaled, on the
luma and chroma planes of the decoded frame ([ADR-0004](../adr/0004-irreversible-blur.md)).
What remains carries no recoverable detail: there is no light Gaussian blur
that deconvolution could undo. `BLUR_METHOD=solid` fills regions with a flat
colour instead.

## What is stored, where, and for how long

| Data | Where | Deleted when |
|---|---|---|
| Original upload | `DATA_DIR/jobs/<id>/input.*` | As soon as rendering ends, on any failure and on cancel. The janitor removes leftovers of crashed workers. |
| `detections.jsonl` (box coordinates, no pixels) | Job folder | With the job's results. |
| Blurred video, metadata, best-frame pictures | Job folder | `RESULT_TTL_MINUTES` after success (default 60), or on `DELETE /jobs/<id>`. Synchronous (`sync=1`) results right after they are sent. |
| `keep=1` regions | `KEEP_DIR`, encrypted | `KEEP_TTL_HOURS` (default 48). |
| Job row (id, timings, counters, error code) | SQLite job store | One day after the job ends. No file name, no user data. |

Only the blurred outputs leave the service. The CLI writes its outputs where
you tell it to and keeps nothing else, except temporary files deleted at the
end of the command.

## `keep=1`: encrypted regions with a lifetime

Panoramax can ask a blurring service to keep the original pixels of blurred
regions, so that a false positive (a sign blurred as a plate) can be
un-blurred later. SGBlur-Video only keeps regions of chains whose best score is
below `KEEP_MAX_CONFIDENCE` (likely false positives). Unlike SGBlur, these
crops are:

- encrypted with AES-256-GCM, with a key derived from the server secret
  `KEEP_SECRET_KEY` and the job's `blurring_id` (neither alone can read them);
- stored under a name that cannot be linked to the job (SHA-256 of the
  `blurring_id`);
- deleted after `KEEP_TTL_HOURS`.

`keep=1` is refused when `KEEP_SECRET_KEY` is not set. The route that would
un-blur from these archives is planned for v2.

## What is never logged

Logs contain job ids, phases, counts, timings and error codes. They never
contain upload file names, paths of uploaded files, coordinates of detections,
pictures or thumbnails. Error messages returned by the API follow the same
rule.

## Known limits

The blur is only as good as the detector. It misses more often:

- very small faces and plates (less than about 15 px high), far from the camera;
- faces seen from behind or heavily occluded, motion blur, night footage;
- unusual plates (motorbikes, foreign formats) and faces in reflections;
- the `fast` detection profile on high-resolution video: it found about three
  times fewer faces than `standard` on 8K footage
  ([benchmarks](../guides/benchmarks.md)).

What the project measures:

- every push runs a synthetic privacy test on the whole pipeline down to the
  encoded file, which must leak nothing;
- the privacy benchmark measures leakage on manually annotated real clips
  ([testing strategy](../design/testing-strategy.md)). Its results on real
  footage are pending: review outputs before publishing them, especially
  crowded scenes.

## Reporting a privacy leak

If you find a face or a plate left visible, report it with the privacy-leak
issue template, **without attaching the video or any identifying detail**:
see [SECURITY.md](https://github.com/KylianAUBRY/SGBlur_Video/blob/main/SECURITY.md).
