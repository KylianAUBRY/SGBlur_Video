# Security and privacy policy

SGBlur-Video exists to protect the privacy of people filmed in public space.
We treat **privacy leaks as security vulnerabilities**.

## What to report

- A face or a licence plate left visible (even on a single frame) in an output video.
- A blur that can be reversed or seen through.
- Original pixels, original videos or embedded previews surviving in outputs, temporary files, logs or `keep=1` storage longer than documented.
- Any classic vulnerability: authentication bypass, path traversal, SSRF through `callback_url`, denial of service through crafted videos, etc.

## How to report (v1)

SGBlur-Video v1 is developed during a time-limited hackathon and is
**alpha: it must not be deployed in production**. Until v2, reports go
through a **public issue** with the
[**Privacy leak / security** template](https://github.com/KylianAUBRY/SGBlur_Video/issues/new?template=privacy_leak.yml),
under strict rules:

- **Never attach** the video, a picture, a screenshot or a link to media showing the person or the plate.
- **Never give** a place, a date, a name or anything that could identify the person, the vehicle or where the video was shot.
- **Describe** the technical situation instead: resolution, frame rate, 360° or not, approximate object size in pixels, motion, lighting, how many frames stay visible, version and settings (`sgblur-video version`, `sgblur-video config`, which shows your home folder as `~`; remove any other path that could identify you).

Maintainers delete any issue or comment that breaks these rules. If they need
media to reproduce a leak, they agree with the reporter on a private, temporary
channel and delete the media afterwards.

Confirmed privacy leaks are fixed with priority and become a regression test
(synthetic when possible).

## Planned for v2: private reporting

v2 will enable [GitHub private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability),
so that reports reach only the maintainers. This page, the issue templates and
the code of conduct will then point to it.

## Supported versions

The project is alpha: only the `main` branch is supported.
