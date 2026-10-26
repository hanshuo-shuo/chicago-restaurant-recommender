"""Draw docs/images/architecture.svg and refresh the copy inlined in the web UI.

    python docs/architecture_diagram.py            # or: make diagram

The drawing is two swimlanes: offline training above (ingestion, split, features, retrieval models, ranker,
evaluation, versioned artifacts) and online serving below (request, candidates, rules, ranker, optional LLM arm,
response, impression log, per-arm CTR and the position-debiased feedback ranker). Geometry is computed here so
boxes and arrows stay aligned when labels change.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SVG_PATH = ROOT / "docs" / "images" / "architecture.svg"
HTML_PATH = ROOT / "src" / "restaurant_recs" / "static" / "index.html"

W, H = 1200, 664
FONT = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Drawing:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def box(self, x, y, w, h, title, sub=None, cls="ad-box", extra="", sub2=None) -> None:
        self.add(f'<rect class="{cls}{(" " + extra) if extra else ""}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>')
        cx = x + w / 2
        lines = [line for line in (sub, sub2) if line]
        if not lines:
            self.add(f'<text class="ad-t" x="{cx}" y="{y + h / 2 + 4.5}" text-anchor="middle">{esc(title)}</text>')
            return
        top = y + h / 2 - 6.5 * len(lines) + 1.5
        self.add(f'<text class="ad-t" x="{cx}" y="{top:.1f}" text-anchor="middle">{esc(title)}</text>')
        for i, line in enumerate(lines):
            self.add(f'<text class="ad-s" x="{cx}" y="{top + 15 + 13 * i:.1f}" text-anchor="middle">{esc(line)}</text>')

    def edge(self, d, arrow=True, extra="") -> None:
        marker = ' marker-end="url(#ad-arrow)"' if arrow else ""
        self.add(f'<path class="ad-edge{(" " + extra) if extra else ""}" d="{d}"{marker}/>')

    def label(self, x, y, text, anchor="start", cls="ad-el") -> None:
        self.add(f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{esc(text)}</text>')


def render() -> str:
    d = Drawing()
    d.add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
          'aria-label="System flow: offline ingestion, features, retrieval models and a LightGBM ranker produce versioned '
          'artifacts; online, a request goes through candidate generation, business rules, the ranker and an optional LLM '
          'arm; impressions and events are logged and feed per-arm CTR and a position-debiased ranker.">')
    d.add("<style>")
    d.add(f".ad-t{{font:600 12.5px {FONT};fill:#1B2733}}")
    d.add(f".ad-s{{font:400 10.5px {FONT};fill:#5C6B7A}}")
    d.add(f".ad-lab{{font:700 10.5px {FONT};fill:#2B6CB0;letter-spacing:1.6px}}")
    d.add(f".ad-lab2{{font:400 10.5px {FONT};fill:#5C6B7A}}")
    d.add(f".ad-el{{font:400 10.5px {FONT};fill:#4A5A6A}}")
    d.add(".ad-bg{fill:#FFFFFF}.ad-lane{fill:#F2F5F9}.ad-lane2{fill:#F7F7F4}")
    d.add(".ad-box{fill:#FFFFFF;stroke:#C5CFDA;stroke-width:1.2}")
    d.add(".ad-model{fill:#E6F0FA;stroke:#2B6CB0;stroke-width:1.2}")
    d.add(".ad-store{fill:#EEF1F4;stroke:#9AA7B4;stroke-width:1.2}")
    d.add(".ad-group{fill:#FBFCFE;stroke:#2B6CB0;stroke-width:1;stroke-dasharray:3 3}")
    d.add(".ad-opt{stroke-dasharray:5 4}")
    d.add(".ad-edge{fill:none;stroke:#4A5A6A;stroke-width:1.3}")
    d.add(".ad-div{stroke:#B9C4CF;stroke-width:1;stroke-dasharray:4 4}")
    d.add("</style>")
    d.add('<defs><marker id="ad-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
          'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#4A5A6A"/></marker></defs>')
    d.add(f'<rect class="ad-bg" x="0" y="0" width="{W}" height="{H}" rx="14"/>')

    # offline lane
    d.add('<rect class="ad-lane" x="12" y="14" width="1176" height="300" rx="12"/>')
    d.label(28, 40, "OFFLINE", cls="ad-lab")
    d.label(96, 40, "training, selection on validation, one scoring of the test split  ·  make data train", cls="ad-lab2")
    ya, ha = 96, 58
    d.box(24, ya, 150, ha, "Google Local (2021)", "23M Illinois ratings")
    d.box(204, ya, 160, ha, "Ingest & validate", "scope · dedupe · 5/10-core")
    d.box(394, ya, 150, ha, "Chronological split", "train · validation · test")
    d.box(574, ya, 170, ha, "Item features", "MiniLM text · categories · geo")
    d.add('<rect class="ad-group" x="774" y="52" width="200" height="154" rx="10"/>')
    d.label(874, 70, "RETRIEVAL MODELS", anchor="middle", cls="ad-lab")
    d.box(786, 80, 176, 34, "Two-tower · in-batch + logQ", cls="ad-model")
    d.box(786, 122, 176, 34, "Semantic IDs · RQ-KMeans", cls="ad-model")
    d.box(786, 164, 176, 34, "Sequential Transformer", cls="ad-model")
    d.box(1004, ya, 172, ha, "LightGBM ranker", "14 features · loss chosen on val", cls="ad-model")
    yb = 232
    d.box(774, yb, 200, 50, "Offline evaluation", "full catalog · NDCG@10 · 95% CI")
    d.box(1004, yb, 172, 50, "Versioned artifacts", "data fingerprint · model version", cls="ad-store")
    cy = ya + ha / 2
    for x0, x1 in ((174, 204), (364, 394), (544, 574), (744, 774), (974, 1004)):
        d.edge(f"M{x0},{cy} H{x1}")
    d.edge(f"M874,206 V{yb}")
    d.edge(f"M1090,{ya + ha} V{yb}")
    d.label(1082, 222, "weights · trees · report", anchor="end")

    d.add('<line class="ad-div" x1="12" y1="322" x2="1188" y2="322"/>')

    # online lane
    d.add('<rect class="ad-lane2" x="12" y="330" width="1176" height="322" rx="12"/>')
    d.label(28, 354, "ONLINE", cls="ad-lab")
    d.label(90, 354, "serving, logging and learning from feedback  ·  make serve", cls="ad-lab2")
    d.add('<rect class="ad-group" x="194" y="366" width="790" height="104" rx="10"/>')
    d.label(974, 382, "INFERENCE ENGINE  ·  NumPy + LightGBM, no PyTorch at serving time", anchor="end", cls="ad-lab")
    yc, hc = 390, 58
    d.box(24, yc, 150, hc, "POST /api/recommend", "cuisines · location · price · seeds")
    d.box(204, yc, 160, hc, "Candidate generation", "4 sources · ≈220 candidates")
    d.box(394, yc, 150, hc, "Business rules", "cuisine · distance · price · hours")
    d.box(574, yc, 170, hc, "LightGBM ranker", "fallback: fixed blend", cls="ad-model")
    d.box(774, yc, 200, hc, "LLM rerank · optional arm", "2 orders · Borda merge · fallback", extra="ad-opt")
    d.box(1004, yc, 172, hc, "Top-k + explanations", "model version · latency")
    cyc = yc + hc / 2
    for x0, x1 in ((174, 204), (364, 394), (544, 574), (744, 774), (974, 1004)):
        d.edge(f"M{x0},{cyc} H{x1}")
    d.edge("M1090,282 V340 H880 V366")
    d.label(1076, 335, "loaded at startup, verified against the data fingerprint", anchor="end")

    # feedback row
    d.box(24, 528, 150, 56, "POST /api/events", "click · favorite")
    d.box(204, 506, 300, 100, "Impression log · SQLite", "request · position · scores · model version",
          sub2="events accepted only against a logged impression", cls="ad-store")
    d.box(574, 506, 170, 44, "Debiased feedback ranker", "MMoE + position tower", cls="ad-model")
    d.box(574, 562, 170, 44, "Per-arm CTR · favorite rate", "A/B readout")
    d.edge("M174,556 H204")
    d.edge("M504,556 H540 V528 H574")
    d.edge("M540,556 V584 H574")
    d.edge("M1090,448 V626 H354 V606")
    d.label(720, 620, "one impression row per returned item", anchor="middle")
    d.edge("M659,506 V470", extra="ad-opt")
    d.label(668, 492, "swaps in once real traffic reaches 2k impressions")
    d.add("</svg>")
    return "\n".join(d.parts) + "\n"


def main() -> None:
    svg = render()
    SVG_PATH.write_text(svg, encoding="utf-8")
    inline = svg.strip().replace(' width="1200" height="664"', "", 1)
    html = HTML_PATH.read_text(encoding="utf-8")
    html, n = re.subn(r'(<div class="diagram-scroll">)(.*?)(</div>)', lambda m: m.group(1) + inline + m.group(3), html, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("index.html has no <div class=\"diagram-scroll\"> to refresh")
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {SVG_PATH.relative_to(ROOT)} and refreshed the inline copy in {HTML_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
