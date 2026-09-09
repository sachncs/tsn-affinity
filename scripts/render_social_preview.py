"""Generate the GitHub social-preview banner.

The banner is 1280 x 640 (the recommended size for a GitHub social
preview image) and ships under ``docs/assets/images/social-preview.png``.
It uses a teal-to-amber gradient with the brand name and tagline.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont


def _resolve_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Return a font of the given size, falling back to PIL defaults."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Black.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _gradient(width: int, height: int) -> Image.Image:
    """Build a horizontal teal-to-amber gradient with subtle blur."""
    start = (0, 105, 92)    # teal 800
    end = (255, 179, 0)     # amber 600
    grad = Image.new("RGB", (width, height), start)
    draw = ImageDraw.Draw(grad)
    for x in range(width):
        ratio = x / max(1, width - 1)
        r = int(start[0] + (end[0] - start[0]) * ratio)
        g = int(start[1] + (end[1] - start[1]) * ratio)
        b = int(start[2] + (end[2] - start[2]) * ratio)
        draw.line([(x, 0), (x, height)], fill=(r, g, b))
    return grad.filter(ImageFilter.GaussianBlur(radius=2))


def render(path: str) -> None:
    """Render the banner to ``path``."""
    width, height = 1280, 640
    image = _gradient(width, height)
    draw = ImageDraw.Draw(image)

    # Decorative shapes representing sparse subnetwork masks.
    for offset in (80, 240, 400):
        for x in range(0, width, 80):
            for y in range(0, height, 80):
                cx = x + (offset // 4)
                cy = y + ((offset // 2) % 60)
                size = 8 + ((x + y + offset) % 16)
                alpha = 32 + ((x * y + offset) % 96)
                overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                od = ImageDraw.Draw(overlay)
                od.ellipse(
                    (cx - size, cy - size, cx + size, cy + size),
                    fill=(255, 255, 255, alpha),
                )
                image = Image.alpha_composite(image.convert("RGBA"), overlay)
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    title_font = _resolve_font(96, bold=True)
    subtitle_font = _resolve_font(34, bold=False)
    badge_font = _resolve_font(20, bold=False)

    title = "TSN-Affinity"
    subtitle = "Continual Offline RL with Sparse Subnetworks & Affinity Routing"

    # Title shadow for legibility against the gradient.
    draw.text((81, 191), title, font=title_font, fill=(0, 0, 0, 90))
    draw.text((80, 190), title, font=title_font, fill=(255, 255, 255))
    draw.text((81, 301), subtitle, font=subtitle_font, fill=(0, 0, 0, 70))
    draw.text((80, 300), subtitle, font=subtitle_font, fill=(255, 255, 255))

    # Footer badges.
    badges = [
        "pip install tsn-affinity",
        "Python 3.10+",
        "MIT License",
        "sachncs.github.io/tsn-affinity",
    ]
    x = 80
    y = height - 80
    for badge in badges:
        bbox = draw.textbbox((0, 0), badge, font=badge_font)
        padding = 16
        w = bbox[2] - bbox[0] + padding * 2
        h = bbox[3] - bbox[1] + padding
        draw.rounded_rectangle(
            (x, y, x + w, y + h), radius=h // 2, fill=(0, 0, 0, 110)
        )
        draw.text((x + padding, y + padding // 2), badge, font=badge_font, fill=(255, 255, 255))
        x += w + 16

    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, "PNG", optimize=True)
    print(f"Wrote social preview: {path}")


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "docs/assets/images/social-preview.png"
    render(target)
