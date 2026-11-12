"""spectral-trust logo, mark and favicon: a cycle cut in two by its Fiedler partition (blue and orange, the
library's Fiedler colour map), tilted 28 degrees, with an Inter SemiBold wordmark converted to paths.

    python docs/make_logo.py

Writes docs/assets/logo.svg, logo-dark.svg, mark.svg, mark-dark.svg and favicon.svg. Inter (SIL Open Font
Licence) is fetched once from Google Fonts into docs/.cache/ if it is not there.
"""
import math
import pathlib
import re
import urllib.request

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "assets"
CACHE = ROOT / ".cache"
BLUE, ORANGE = "#0072B2", "#E69F00"
TILT = math.radians(28.0)


def inter_semibold() -> pathlib.Path:
    path = CACHE / "Inter-SemiBold.ttf"
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        css = urllib.request.urlopen("https://fonts.googleapis.com/css2?family=Inter:wght@600").read().decode()
        url = re.search(r"https://[^)]+\.ttf", css).group(0)
        path.write_bytes(urllib.request.urlopen(url).read())
    return path

def arc(cx, cy, r, a0, a1, n=120):
    pts = [(cx + r * math.cos(a0 + (a1 - a0) * i / n), cy + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]
    return "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts)


def mark(ink, cx=32.0, cy=32.0, R=21.0, w=6.0, node=4.6, gap_deg=16):
    g = math.radians(gap_deg)
    top, bot = -math.pi / 2 + TILT, math.pi / 2 + TILT
    p = [
        f'<path d="{arc(cx, cy, R, top + g, bot - g)}" fill="none" stroke="{ORANGE}" stroke-width="{w}" stroke-linecap="round"/>',
        f'<path d="{arc(cx, cy, R, bot + g, top + 2 * math.pi - g)}" fill="none" stroke="{BLUE}" stroke-width="{w}" stroke-linecap="round"/>',
    ]
    for a in (top, bot):
        p.append(f'<circle cx="{cx + R * math.cos(a):.2f}" cy="{cy + R * math.sin(a):.2f}" r="{node}" fill="{ink}"/>')
    return "".join(p)


def wordmark(text, x, baseline, size, tracking=-0.025):
    font = TTFont(inter_semibold())
    gs, cmap, hm = font.getGlyphSet(), font.getBestCmap(), font["hmtx"]
    s = size / font["head"].unitsPerEm
    paths, px = [], x
    for ch in text:
        name = cmap[ord(ch)]
        pen = SVGPathPen(gs)
        gs[name].draw(TransformPen(pen, (s, 0, 0, -s, px, baseline)))
        if pen.getCommands():
            paths.append(pen.getCommands())
        px += hm[name][0] * s + tracking * size
    return " ".join(paths), px


for theme, ink in (("light", "#14161A"), ("dark", "#F4F5F7")):
    sfx = "" if theme == "light" else "-dark"
    m = mark(ink)
    d, end = wordmark("spectral-trust", x=74, baseline=43.5, size=32)
    W = math.ceil(end + 4)
    (OUT / f"logo{sfx}.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 64" width="{W}" height="64" role="img" '
        f'aria-label="spectral-trust"><title>spectral-trust</title>{m}<path d="{d}" fill="{ink}"/></svg>', encoding="utf-8")
    (OUT / f"mark{sfx}.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64" role="img" '
        f'aria-label="spectral-trust"><title>spectral-trust</title>{m}</svg>', encoding="utf-8")
# favicon: thicker strokes for 16 px, adapts to the browser theme
light, dark = mark("#14161A", w=7.5, node=5.6), mark("#F4F5F7", w=7.5, node=5.6)
(OUT / "favicon.svg").write_text(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><style>.d{display:none}'
    '@media (prefers-color-scheme: dark){.l{display:none}.d{display:inline}}</style>'
    f'<g class="l">{light}</g><g class="d">{dark}</g></svg>', encoding="utf-8")
print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
