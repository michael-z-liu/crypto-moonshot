"""Train the day5 variant's 2026 fold (5 seeds, day-weighted path CNN; trained before 2025-07-01,
early-stopped on 2025 H2) and save its weights and reference scores for forward scoring, in the
layout `paper.Model` reads (seed{j}.pt, norm.npz with mu/sd/liq_mean/liq_sd/ref_sorted).

Run (GPU, ~15-25 min): uv run --group research python save_day_model.py [variant]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["MOONSHOT_UNIVERSE"] = "all"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "volume_spike"))

import i10_train as i10  # noqa: E402
import m12_cnn_wf as m12  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "day5"
OUT = HERE.parents[1] / ".cache" / "forward_model" / VARIANT
PAPER_NORM = HERE.parents[1] / ".cache" / "paper" / "model" / "norm.npz"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    saved: dict[str, np.ndarray] = {}

    def hook(label: str, seed: int | None, net: object, ref: np.ndarray | None) -> None:
        if label != "2026":
            return
        if net is not None:
            torch.save(net.state_dict(), OUT / f"seed{seed}.pt")  # type: ignore[attr-defined]
        if ref is not None:
            saved["ref_sorted"] = ref

    i10.SAVE_HOOK = hook
    icfg = i10.I10[VARIANT]
    data = m12.Data()
    i10.run(f"fwd_{VARIANT}_2026", data, icfg, years=(2026,))
    norm = dict(np.load(PAPER_NORM))
    assert np.allclose(norm["mu"], data.mu), "frozen channel means differ from the paper model's"
    assert np.allclose(norm["sd"], data.sd), "frozen channel sds differ from the paper model's"
    norm["ref_sorted"] = saved["ref_sorted"]
    np.savez(OUT / "norm.npz", **norm)
    (OUT / "config.json").write_text(
        json.dumps(
            {"variant": VARIANT, "fold": "2026", "seeds": icfg.base.seeds, "cfg": icfg.describe()},
            indent=1,
        )
    )
    print("saved", OUT, file=sys.stderr)


if __name__ == "__main__":
    main()
