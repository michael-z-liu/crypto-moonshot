"""M3: the *every candidate* rows of the ledger (no model).

    uv run --group research python m3_every.py

Scores the limit family at buffer 0.05u with exits E1..E6 (E1 is the base rule) and touch-market
with E1, every candidate taken, 10 slots. Each call upserts its ledger row and writes its sealed
file; the 2022-24 cards are printed and saved to `results/m3_every.txt`.
"""

from __future__ import annotations

import time

from common import EXITS, RESULTS
from scorecard import Rule, scorecard, show

STAGE = "M3"
SOURCE = "every candidate (no model)"


def rules() -> list[tuple[str, Rule]]:
    out = [(f"every-L0.05-{e}", Rule(family="limit", buffer=0.05, exit=e)) for e in EXITS]
    out.append(("every-touch-E1", Rule(family="touch", exit="E1")))
    return out


def main() -> None:
    texts = []
    for name, rule in rules():
        t0 = time.perf_counter()
        card = scorecard(name, rule, stage=STAGE, source=SOURCE)
        text = show(card)
        print(text, flush=True)
        print(f"  ({time.perf_counter() - t0:.1f}s)\n", flush=True)
        texts.append(text)
    (RESULTS / "m3_every.txt").write_text("\n\n".join(texts) + "\n")


if __name__ == "__main__":
    main()
