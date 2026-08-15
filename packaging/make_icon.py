"""Draw the app icon as a PNG.

Run at install time; the installer converts the result to .icns with the
iconutil tool that ships with macOS. Kept dependency-light on purpose: Pillow
only, no design assets to ship.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 1024
BG_TOP = (180, 83, 9)          # warm amber
BG_BOTTOM = (120, 53, 15)
INK = (255, 251, 235)


def rounded_mask(size: int, radius_ratio: float = 0.2237) -> Image.Image:
    """macOS "squircle"-ish corner radius."""
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, size - 1, size - 1],
                        radius=int(size * radius_ratio), fill=255)
    return mask


def gradient(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(img)
    for y in range(size):
        t = y / max(1, size - 1)
        d.line([(0, y), (size, y)],
               fill=tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)))
    return img


def draw(size: int = SIZE) -> Image.Image:
    img = gradient(size).convert("RGBA")
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    cx, cy = size / 2, size / 2

    # One waveform read left to right: the incoming language fades out as the
    # new voice comes in. Faint bars on the left, solid on the right.
    bars = 9
    spacing = size * 0.079
    left = cx - spacing * (bars - 1) / 2
    w = size * 0.044

    for i in range(bars):
        t = i / (bars - 1)
        # Speech-like envelope: uneven, not a symmetrical level meter.
        env = (0.34 * math.sin(t * math.pi * 2.7 + 0.6)
               + 0.30 * math.sin(t * math.pi * 1.3)
               + 0.52)
        env = max(0.16, min(1.0, env))
        h = size * 0.30 * env

        # Opacity ramps from faint to solid across the middle third.
        alpha = int(255 * (0.22 + 0.78 * min(1.0, max(0.0, (t - 0.18) / 0.5))))
        x = left + i * spacing
        d.rounded_rectangle([x - w / 2, cy - h, x + w / 2, cy + h],
                            radius=w / 2, fill=(*INK, alpha))

    img = Image.alpha_composite(img, overlay)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img, (0, 0), rounded_mask(size))
    return out


if __name__ == "__main__":
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else "icon.png")
    draw().save(dest)
    print(dest)
