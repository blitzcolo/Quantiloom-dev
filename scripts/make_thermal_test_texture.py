#!/usr/bin/env python3
"""Generate the temperature map assets/textures/thermal_gradient.png.

The map is the provenance of a test asset rather than a tool anybody runs
regularly: a horizontal ramp is the one pattern whose render can be checked
without a reference image, because the radiance along a row must increase
monotonically and by a known amount.

The byte value IS the normalised temperature. Quantiloom reads a PNG by
dividing bytes by 255 with no sRGB decode, so what is written here is what the
shader decodes as T = value * temperature_scale + temperature_offset. Do not
produce this file with a paint program set to a colour profile.

stdlib only, matching the rest of scripts/.
"""

import struct
import zlib
from pathlib import Path

WIDTH = 256
HEIGHT = 256
OUT = Path(__file__).resolve().parent.parent / "assets" / "textures" / "thermal_gradient.png"


def chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def main() -> None:
    # One byte per texel, greyscale. Column x holds x/(W-1) of the range, so
    # with scale = 60 and offset = 270 the plate runs 270 K to 330 K left to
    # right, and each step is 60/255 = 0.24 K.
    rows = bytearray()
    for _y in range(HEIGHT):
        rows.append(0)  # PNG filter type 0 (None) for this scanline
        for x in range(WIDTH):
            rows.append(round(x * 255 / (WIDTH - 1)))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(rows), 9))
    png += chunk(b"IEND", b"")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(png)
    print(f"wrote {OUT} ({WIDTH}x{HEIGHT}, {len(png)} bytes)")


if __name__ == "__main__":
    main()
