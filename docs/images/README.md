# Icons and banners

Same family as neurosuite-3's (`doc/images/README.md` there): a rounded navy tile, and a cyan
recording trace that runs off both edges of the tile with one extracellular spike in it.  Each icon
puts its own motif above the trace.

| Icon | Motif | Source |
|---|---|---|
| fiber-kit | three fibers (cyan, white, orange) fanning out from one low-energy origin, spikes riding each curve and growing with energy -- a unit traces one curve d(r) | `fiber-kit.svg` |
| Fibers (banner chip) | one fiber, spikes sliding along it | `fibers.svg` |
| Linking (banner chip) | one cell's fiber cut into three chunks (dashed boundaries), the pieces drifting, joined across the cuts | `linking.svg` |
| Neurons (banner chip) | two fibers inside one cell outline | `neurons.svg` |
| The pipeline (doc page) | four steps in a row | `pipeline.svg` |
| Stages & parameters (doc page) | a prompt above a flag table | `stages.svg` |
| Core methods (doc page) | a fiber on axes, with its tangent at one spike | `methods.svg` |
| neurosuite-3 (banner kicker) | copied from neurosuite-3 `doc/images/neurosuite-3.svg` | `neurosuite-3.svg` |

Palette: tile gradient `#0E2236` -> `#173F5C`, cyan `#43C8E2`, white `#EEF4F8`, orange `#FF8F45`.
The SVGs are 512 x 512, shapes and strokes only (no fonts).

Rendered from them by `render.py` (Chromium via Playwright, Inter font; `pngquant` when installed):

- `banner.png` (2560 x 800, shown at half size): the top of the README; `banner-narrow.png`
  (1600 x 680: icon, name, a short line) replaces it below 600 px;
- `social-preview.png` (1280 x 640): GitHub's link preview, uploaded by hand under
  Settings -> General -> Social preview;
- `doc-<page>-banner.png` (2560 x 640) and `doc-<page>-banner-narrow.png` (1600 x 600) for
  `pipeline`, `stages` and `methods`: the top of those documentation pages, with a "fiber-kit"
  kicker line, the title and a tagline;
- `icons/hi<size>-app-fiber-kit.png` (16, 22, 32, 48, 64, 128, 256 px) and
  `icons/app-fiber-kit.ico` (16, 24, 32, 48, 64, 256 px), each size rendered from the SVG.

After changing an SVG, run `python3 docs/images/render.py` and look at the 16 px icon: below 32 px
only the motif's colours survive.
