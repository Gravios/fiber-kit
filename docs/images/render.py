#!/usr/bin/env python3
"""Render fiber-kit's README and documentation banners and its app icons from the SVGs in this folder.

Same family as neurosuite-3 (doc/images/README.md there): navy panel, the
icon tile on the left, a cyan recording trace with one spike running across
the bottom.  Rendered with Chromium via Playwright, transparent outside the
panel's rounded corners.

    python3 docs/images/render.py        # banner.png, banner-narrow.png, social-preview.png,
                                          # doc-<page>-banner(-narrow).png, icons/

Needs `playwright` (Chromium) and the Inter font installed; `pngquant`
is used when present to shrink the PNGs.
"""
from __future__ import annotations

import pathlib

HERE = pathlib.Path(__file__).resolve().parent
C, W, O = "#43C8E2", "#EEF4F8", "#FF8F45"


def svg(name: str) -> str:
    s = (HERE / f"{name}.svg").read_text()
    # unique gradient id per inline copy, so several icons can share a page
    s = s.replace('width="512" height="512"', 'width="100%" height="100%"', 1)
    return s.replace('id="bg"', f'id="bg-{name}"').replace("url(#bg)", f"url(#bg-{name})")


def trace(w: int, y: float, spike_x: float, kx: float = 2.2, ky: float = 1.65,
          sw: float = 15) -> str:
    """The icons' spike (bump, deep trough, short low rebound), scaled."""
    def p(dx, dy):
        return f"{spike_x + dx * kx:.1f},{y + dy * ky:.1f}"
    d = (f"M-20,{y} H{spike_x:.1f} L{p(14,-12)} L{p(38,70)} L{p(60,-16)} "
         f"Q{p(76,8)} {p(94,4)} H{w + 20}")
    return (f'<path d="{d}" fill="none" stroke="{C}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')


def name_html() -> str:
    return (f'<span style="color:{C}">fiber</span><span style="color:{W}">-</span>'
            f'<span style="color:{O}">kit</span>')


CSS = f"""
* {{ margin:0; padding:0; box-sizing:border-box; }}
html, body {{ background: transparent; }}
body {{ font-family: 'Inter', sans-serif; -webkit-font-smoothing: antialiased; }}
.panel {{ position:relative; overflow:hidden;
  background: linear-gradient(100deg, #0E2236 0%, #13304A 45%, #173F5C 100%); }}
.tile svg {{ width:100%; height:100%; display:block; }}
.tile {{ position:absolute; filter: drop-shadow(0 18px 40px rgba(0,0,0,.35)); }}
.kicker {{ display:flex; align-items:center; font-weight:700; letter-spacing:.01em; }}
.kicker svg {{ display:block; }}
.title {{ font-weight:800; letter-spacing:-0.02em; line-height:1; }}
.tag {{ color:{W}; opacity:.88; font-weight:400; line-height:1.3; }}
.chips {{ display:flex; align-items:center; }}
.chip {{ display:flex; align-items:center; color:{W}; font-weight:500; }}
.chip svg {{ display:block; border-radius:22%; }}
"""

KICKER_NS3 = (f'<span style="color:{C}">neuro</span><span style="color:{W}">suite</span>'
              f'<span style="color:{O}">-3</span>')


def page(body: str, w: int, h: int, radius: int) -> str:
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style>'
            f'</head><body><div class="panel" style="width:{w}px;height:{h}px;'
            f'border-radius:{radius}px">{body}</div></body></html>')


def banner() -> str:                      # 2560 × 800, like neurosuite-3's
    w, h = 2560, 800
    chips = "".join(
        f'<div class="chip" style="margin-right:64px">'
        f'<div style="width:108px;height:108px;margin-right:30px">{svg(n)}</div>'
        f'<span style="font-size:50px">{label}</span></div>'
        for n, label in (("fibers", "Fibers"), ("linking", "Linking"),
                         ("neurons", "Neurons")))
    body = (
        f'<svg style="position:absolute;left:0;top:0" width="{w}" height="{h}">'
        f'{trace(w, 672, 1520)}</svg>'
        f'<div class="tile" style="left:168px;top:112px;width:472px;height:472px">'
        f'{svg("fiber-kit")}</div>'
        f'<div style="position:absolute;left:752px;top:118px">'
        f'<div class="kicker" style="font-size:46px">'
        f'<div style="width:58px;height:58px;margin-right:20px">{svg("neurosuite-3")}</div>'
        f'{KICKER_NS3}</div>'
        f'<div class="title" style="font-size:156px;margin-top:22px">{name_html()}</div>'
        f'<div class="tag" style="font-size:52px;margin-top:34px">'
        f'Drift-stable reorganization of over-split spike sorts</div>'
        f'<div class="chips" style="margin-top:44px">{chips}</div></div>')
    return page(body, w, h, 52)


def narrow() -> str:                      # 1600 × 680: icon, name, a short line
    w, h = 1600, 680
    body = (
        f'<svg style="position:absolute;left:0;top:0" width="{w}" height="{h}">'
        f'{trace(w, 572, 1060, kx=1.9, ky=1.4, sw=13)}</svg>'
        f'<div class="tile" style="left:110px;top:120px;width:340px;height:340px">'
        f'{svg("fiber-kit")}</div>'
        f'<div style="position:absolute;left:520px;top:150px">'
        f'<div class="kicker" style="font-size:40px">'
        f'<div style="width:50px;height:50px;margin-right:16px">{svg("neurosuite-3")}</div>'
        f'{KICKER_NS3}</div>'
        f'<div class="title" style="font-size:118px;margin-top:20px">{name_html()}</div>'
        f'<div class="tag" style="font-size:46px;margin-top:28px">'
        f'Drift-stable spike-sort reorganization</div></div>')
    return page(body, w, h, 44)


def social() -> str:                      # 1280 × 640: GitHub link preview
    w, h = 1280, 640
    body = (
        f'<svg style="position:absolute;left:0;top:0" width="{w}" height="{h}">'
        f'{trace(w, 540, 830, kx=1.6, ky=1.2, sw=11)}</svg>'
        f'<div class="tile" style="left:80px;top:120px;width:300px;height:300px">'
        f'{svg("fiber-kit")}</div>'
        f'<div style="position:absolute;left:430px;top:150px">'
        f'<div class="kicker" style="font-size:32px">'
        f'<div style="width:40px;height:40px;margin-right:14px">{svg("neurosuite-3")}</div>'
        f'{KICKER_NS3}</div>'
        f'<div class="title" style="font-size:92px;margin-top:16px">{name_html()}</div>'
        f'<div class="tag" style="font-size:36px;margin-top:24px">'
        f'Drift-stable reorganization<br>of over-split spike sorts</div></div>')
    return page(body, w, h, 0)


KICKER_FK = name_html


# Documentation pages: (key, icon svg, title, tagline, narrow tagline).
DOC_PAGES = [
    ("pipeline", "pipeline", "The pipeline",
     "How the stages fit together, and how to run them",
     "How the stages fit together"),
    ("stages", "stages", "Stages &amp; parameters",
     "Every fiber-* command, its flags and defaults",
     "Every command and flag"),
    ("methods", "methods", "Core methods",
     "The numerical primitives behind the stages",
     "Primitives behind the stages"),
]


def doc_banner(icon: str, title: str, tag: str) -> str:   # 2560 × 640
    w, h = 2560, 640
    body = (
        f'<svg style="position:absolute;left:0;top:0" width="{w}" height="{h}">'
        f'{trace(w, 553, 1560, kx=2.0, ky=1.5)}</svg>'
        f'<div class="tile" style="left:160px;top:88px;width:392px;height:392px">'
        f'{svg(icon)}</div>'
        f'<div style="position:absolute;left:660px;top:126px">'
        f'<div class="kicker" style="font-size:46px">'
        f'<div style="width:58px;height:58px;margin-right:20px">{svg("fiber-kit")}</div>'
        f'{KICKER_FK()}</div>'
        f'<div class="title" style="font-size:150px;margin-top:20px;color:{W}">{title}</div>'
        f'<div class="tag" style="font-size:52px;margin-top:34px">{tag}</div></div>')
    return page(body, w, h, 52)


def doc_banner_narrow(icon: str, title: str, tag: str) -> str:   # 1600 × 600
    w, h = 1600, 600
    body = (
        f'<svg style="position:absolute;left:0;top:0" width="{w}" height="{h}">'
        f'{trace(w, 518, 940, kx=1.8, ky=1.35, sw=13)}</svg>'
        f'<div class="tile" style="left:104px;top:80px;width:320px;height:320px">'
        f'{svg(icon)}</div>'
        f'<div style="position:absolute;left:500px;top:116px">'
        f'<div class="kicker" style="font-size:42px">'
        f'<div style="width:56px;height:56px;margin-right:18px">{svg("fiber-kit")}</div>'
        f'{KICKER_FK()}</div>'
        f'<div class="title" style="font-size:112px;margin-top:22px;color:{W}">{title}</div>'
        f'<div class="tag" style="font-size:52px;margin-top:30px">{tag}</div></div>')
    return page(body, w, h, 44)


# hicolor sizes and .ico sizes, as in neurosuite-3 (doc/images/README.md there)
ICON_SIZES = (16, 22, 32, 48, 64, 128, 256)
ICO_SIZES = (16, 24, 32, 48, 64, 256)


def main() -> None:
    from playwright.sync_api import sync_playwright
    jobs = [("banner.png", banner(), 2560, 800),
            ("banner-narrow.png", narrow(), 1600, 680),
            ("social-preview.png", social(), 1280, 640)]
    for key, icon, title, tag, ntag in DOC_PAGES:
        jobs.append((f"doc-{key}-banner.png", doc_banner(icon, title, tag), 2560, 640))
        jobs.append((f"doc-{key}-banner-narrow.png",
                     doc_banner_narrow(icon, title, ntag), 1600, 600))
    icon_dir = HERE / "icons"
    icon_dir.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        for out, html, w, h in jobs:
            pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
            pg.set_content(html)
            pg.evaluate("document.fonts.ready")
            pg.screenshot(path=str(HERE / out), omit_background=True)
            print("wrote", out)
        # App icons: every hicolor size rendered straight from the SVG
        # (not downscaled from 256), so small sizes get the renderer's own
        # hinting, as neurosuite-3's were.
        for size in sorted(set(ICON_SIZES) | set(ICO_SIZES)):
            pg = b.new_page(viewport={"width": size, "height": size})
            pg.set_content(f'<html><body style="margin:0;background:transparent">'
                           f'<div style="width:{size}px;height:{size}px">'
                           f'{svg("fiber-kit")}</div></body></html>')
            pg.screenshot(path=str(icon_dir / f"hi{size}-app-fiber-kit.png"),
                          omit_background=True)
        b.close()
    # .ico from the per-size renderings (not a single downscale).
    from PIL import Image
    imgs = [Image.open(icon_dir / f"hi{s}-app-fiber-kit.png").convert("RGBA")
            for s in ICO_SIZES]
    imgs[-1].save(icon_dir / "app-fiber-kit.ico",
                  sizes=[(s, s) for s in ICO_SIZES], append_images=imgs[:-1])
    for s in set(ICO_SIZES) - set(ICON_SIZES):        # 24 px is .ico-only
        (icon_dir / f"hi{s}-app-fiber-kit.png").unlink()
    print("wrote icons/")
    # Dithered 256-colour reduction (~150-250 KB instead of ~1-1.7 MB).
    # A plain PIL octree quantize bands the panel gradient and haloes the
    # tile shadow; pngquant's dithering doesn't.  Skipped if not installed.
    import shutil, subprocess
    if shutil.which("pngquant"):
        for out, *_ in jobs:
            subprocess.run(["pngquant", "--force", "--skip-if-larger",
                            "--quality", "80-100", "--speed", "1",
                            "--output", str(HERE / out), str(HERE / out)],
                           check=False)


if __name__ == "__main__":
    main()
