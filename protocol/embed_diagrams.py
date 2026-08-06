"""Inline the diagram SVGs into the slide deck as data URIs.

The deck is meant to travel as one file: opened from a copy, served by a
viewer that only has the HTML, or mailed on its own. External
`src="diagrams/…"` references break in all three cases, so every diagram is
embedded instead.

Each slide image carries `data-diagram="<file>"`, which is the source of
truth. Re-run this after editing anything in `diagrams/` to refresh the
embedded copies:

    python protocol/embed_diagrams.py
"""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DECK = ROOT / "cv-protocol-slides.html"
DIAGRAMS = ROOT / "diagrams"

IMG = re.compile(r'<img\b[^>]*\bdata-diagram="([^"]+)"[^>]*>')
SRC = re.compile(r'\ssrc="[^"]*"')


def data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{payload}"


def embed(deck: Path = DECK, diagrams: Path = DIAGRAMS) -> dict[str, int]:
    html = deck.read_text(encoding="utf-8")
    report: dict[str, int] = {}

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        source = diagrams / name
        if not source.is_file():
            raise FileNotFoundError(f"missing diagram: {source}")
        uri = data_uri(source)
        report[name] = source.stat().st_size
        tag = SRC.sub("", match.group(0))
        return tag.replace("<img", f'<img src="{uri}"', 1)

    updated = IMG.sub(replace, html)
    if not report:
        raise RuntimeError("no <img data-diagram=…> found; nothing to embed")
    if 'src="diagrams/' in updated:
        raise RuntimeError("an external diagram reference survived")
    deck.write_text(updated, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=DECK)
    parser.add_argument("--diagrams", type=Path, default=DIAGRAMS)
    args = parser.parse_args()
    report = embed(args.deck, args.diagrams)
    for name, size in sorted(report.items()):
        print(f"embedded {name:38} {size:>7,} bytes")
    print(f"{len(report)} diagrams · deck now {args.deck.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
