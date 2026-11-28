"""H1 (holdout): save the day5 variant's OLDEST walk-forward fold (2022: trained on 2020-01 ..
2021-06, early-stopped on 2021 H2, 5 seeds) in the layout `paper.Model` reads, as
`save_day_model.py` does for the 2026 fold. Used as a second scorer of the 2017-19 holdout spikes
(a model trained on the 18 months right after the holdout, instead of 5.5 years).

The fold's walk-forward npz goes to `$DATA/cache/capitulation_ls/holdout/wf/` (not the shared
`wf_i10`); the weights and reference to `.cache/forward_model/day5_2022/`.

Run (GPU, ~10-20 min): uv run --group research python h1_fold2022.py
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

VARIANT = "day5"
FOLD = "2022"
OUT = HERE.parents[1] / ".cache" / "forward_model" / f"{VARIANT}_{FOLD}"
PAPER_NORM = HERE.parents[1] / ".cache" / "forward_model" / VARIANT / "norm.npz"
WF = Path("/Users/mike/code/crypto-moonshot/data/cache/capitulation_ls/holdout/wf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WF.mkdir(parents=True, exist_ok=True)
    i10.WF_I10 = WF
    saved: dict[str, np.ndarray] = {}

    def hook(label: str, seed: int | None, net: object, ref: np.ndarray | None) -> None:
        if label != FOLD:
            return
        if net is not None:
            torch.save(net.state_dict(), OUT / f"seed{seed}.pt")  # type: ignore[attr-defined]
        if ref is not None:
            saved["ref_sorted"] = ref

    i10.SAVE_HOOK = hook
    icfg = i10.I10[VARIANT]
    data = m12.Data()
    i10.run(f"h1_{VARIANT}_{FOLD}", data, icfg, years=(int(FOLD),))
    norm = dict(np.load(PAPER_NORM))
    assert np.allclose(norm["mu"], data.mu), "frozen channel means differ"
    assert np.allclose(norm["sd"], data.sd), "frozen channel sds differ"
    norm["ref_sorted"] = saved["ref_sorted"]
    np.savez(OUT / "norm.npz", **norm)
    (OUT / "config.json").write_text(
        json.dumps(
            {"variant": VARIANT, "fold": FOLD, "seeds": icfg.base.seeds, "cfg": icfg.describe()},
            indent=1,
        )
    )
    # Does the saved fold reproduce day5's own 2022 walk-forward percentiles?
    from common_ls import scores

    new = np.load(WF / f"h1_{VARIANT}_{FOLD}.npz")["pct"]
    old = scores(VARIANT)
    ok = np.isfinite(new)
    same = float(np.mean(new[ok] == old[ok]))
    diff = float(np.max(np.abs(new[ok] - old[ok])))
    print(f"saved {OUT}; 2022 rows {ok.sum()}: equal to day5 {same:.4%}, max diff {diff:.3g}")


if __name__ == "__main__":
    main()
