"""360° (equirectangular) support: wrap-around geometry across the 0°/360° seam.

Projection detection lives in ``core.probe``, circular padding of detection
passes in ``core.detect`` and the spherical metadata transplant in
``core.mp4boxes``; they all use ``wrap``. See ``docs/adr/0007-360-video.md``.
"""
