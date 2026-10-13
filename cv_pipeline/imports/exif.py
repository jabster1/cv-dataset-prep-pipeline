"""Read the EXIF orientation tag (0x0112) from JPEG, PNG and WebP bytes, read-only.

Dependency-free on purpose. Returns 0 when there is no EXIF orientation, else 1..8.
Orientation != 1 means viewers (and OpenCV/Ultralytics decoding) rotate the stored pixels,
so labels authored against un-rotated pixels would not line up.
"""

from __future__ import annotations

import struct


def _tiff_orientation(tiff: bytes) -> int:
    if len(tiff) < 8 or tiff[:2] not in (b"II", b"MM"):
        return 0
    endian = "<" if tiff[:2] == b"II" else ">"
    (ifd,) = struct.unpack(endian + "I", tiff[4:8])
    if ifd + 2 > len(tiff):
        return 0
    (count,) = struct.unpack(endian + "H", tiff[ifd : ifd + 2])
    for i in range(count):
        off = ifd + 2 + 12 * i
        if off + 12 > len(tiff):
            return 0
        tag, typ, _n = struct.unpack(endian + "HHI", tiff[off : off + 8])
        if tag == 0x0112 and typ == 3:
            (value,) = struct.unpack(endian + "H", tiff[off + 8 : off + 10])
            return value if 1 <= value <= 8 else 0
    return 0


def _jpeg(data: bytes) -> int:
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        if marker in (0xD9, 0xDA):  # end of image / start of scan
            return 0
        (length,) = struct.unpack(">H", data[i + 2 : i + 4])
        segment = data[i + 4 : i + 2 + length]
        if marker == 0xE1 and segment[:6] == b"Exif\x00\x00":
            return _tiff_orientation(segment[6:])
        i += 2 + length
    return 0


def _png(data: bytes) -> int:
    i = 8
    while i + 8 <= len(data):
        (length,) = struct.unpack(">I", data[i : i + 4])
        kind = data[i + 4 : i + 8]
        if kind == b"eXIf":
            return _tiff_orientation(data[i + 8 : i + 8 + length])
        if kind == b"IEND":
            return 0
        i += 12 + length
    return 0


def _webp(data: bytes) -> int:
    i = 12
    while i + 8 <= len(data):
        kind = data[i : i + 4]
        (length,) = struct.unpack("<I", data[i + 4 : i + 8])
        if kind == b"EXIF":
            payload = data[i + 8 : i + 8 + length]
            if payload[:6] == b"Exif\x00\x00":
                payload = payload[6:]
            return _tiff_orientation(payload)
        i += 8 + length + (length & 1)
    return 0


def exif_orientation(data: bytes) -> int:
    try:
        if data[:2] == b"\xff\xd8":
            return _jpeg(data)
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return _png(data)
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return _webp(data)
    except struct.error:
        return 0
    return 0
