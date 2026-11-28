"""P1 (perp-only coins): save the day5 variant's walk-forward folds 2023, 2024 and 2025 (5 seeds
each; fold Y trains before Y-1 July and early-stops on Y-1 H2) in the layout `paper.Model` reads,
as `save_day_model.py` does for 2026 and `h1_fold2022.py` for 2022, so that perp-only spikes of
year Y are scored by the fold that never saw year Y (no look-ahead).

Weights and reference scores go to `.cache/forward_model/day5_<Y>/`, the walk-forward npz to
OWN/p1/wf/ (not the shared wf_i10). At the end it checks that the retrained folds reproduce
day5's own walk-forward percentiles on the spot rows of those years.

Run (GPU, ~35-50 min): uv run --group research python p1_folds.py
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
YEARS = (2023, 2024, 2025)
MODELS = HERE.parents[1] / ".cache" / "forward_model"
WF = Path("/Users/mike/code/crypto-moonshot/data/cache/capitulation_ls/p1/wf")


def main() -> None:
    WF.mkdir(parents=True, exist_ok=True)
    i10.WF_I10 = WF
    refs: dict[str, np.ndarray] = {}

    def hook(label: str, seed: int | None, net: object, ref: np.ndarray | None) -> None:
        out = MODELS / f"{VARIANT}_{label}"
        out.mkdir(parents=True, exist_ok=True)
        if net is not None:
            torch.save(net.state_dict(), out / f"seed{seed}.pt")  # type: ignore[attr-defined]
        if ref is not None:
            refs[label] = ref

    i10.SAVE_HOOK = hook
    icfg = i10.I10[VARIANT]
    data = m12.Data()
    name = f"p1_{VARIANT}_" + "_".join(str(y) for y in YEARS)
    i10.run(name, data, icfg, years=YEARS)
    base = dict(np.load(MODELS / VARIANT / "norm.npz"))
    assert np.allclose(base["mu"], data.mu), "frozen channel means differ"
    assert np.allclose(base["sd"], data.sd), "frozen channel sds differ"
    for y in YEARS:
        out = MODELS / f"{VARIANT}_{y}"
        norm = dict(base)
        norm["ref_sorted"] = refs[str(y)]
        np.savez(out / "norm.npz", **norm)
        cfg = {"variant": VARIANT, "fold": str(y), "seeds": icfg.base.seeds, "cfg": icfg.describe()}
        (out / "config.json").write_text(json.dumps(cfg, indent=1))
    from common_ls import meta, scores

    new = np.load(WF / f"{name}.npz")["pct"]
    old = scores(VARIANT)
    yr = meta()["year"].to_numpy()
    for y in YEARS:
        ok = np.isfinite(new) & (yr == y)
        same = float(np.mean(new[ok] == old[ok]))
        diff = float(np.max(np.abs(new[ok] - old[ok])))
        print(f"fold {y}: {ok.sum()} rows, equal to day5 {same:.4%}, max diff {diff:.3g}")


if __name__ == "__main__":
    main()
