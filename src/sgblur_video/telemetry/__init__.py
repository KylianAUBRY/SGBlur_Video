"""Telemetry: GoPro GPMF parsing and GPS position lookup by timestamp.

The GPMF stream itself is preserved by stream copy during rendering; this
package only reads positions to geolocate sign annotations and best frames.
"""
