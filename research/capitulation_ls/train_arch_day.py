"""Train the day-weighted walk-forward path model (the #10 study's `day` setup, 3 seeds) with
another network: a patch transformer (registered here), the dilated TCN or conv+GRU of m12.
Output: cache_all/wf_i10/day_<arch>.npz, comparable with day5 / day10b.

Run: uv run --group research python train_arch_day.py transformer|tcn|gru
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ["MOONSHOT_UNIVERSE"] = "all"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "volume_spike"))

import i10_train as i10
import m12_cnn_wf as m12
import torch
from torch import nn

ARCH = sys.argv[1]


class PatchTransformer(nn.Module):
    """Patches of 6 candles -> tokens, 3 pre-norm encoder layers, CLS token + scalars -> head."""

    def __init__(
        self, ch: int, n_s: int, n_out: int, seq: int, d: int = 64, patch: int = 6
    ) -> None:
        super().__init__()
        self.patch = patch
        n_tok = seq // patch
        self.embed = nn.Linear(ch * patch, d)
        self.pos = nn.Parameter(torch.randn(1, n_tok + 1, d) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(
            d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, 3)
        self.head = m12.Head(d + n_s, n_out)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        p = (
            x.reshape(b, c, n // self.patch, self.patch)
            .permute(0, 2, 1, 3)
            .reshape(b, n // self.patch, c * self.patch)
        )
        z = torch.cat([self.cls.expand(b, -1, -1), self.embed(p)], 1) + self.pos
        return self.head(torch.cat([self.enc(z)[:, 0], s], 1))


m12.ARCHS["transformer"] = lambda cfg, n_s, n_out: PatchTransformer(
    len(cfg.channels), n_s, n_out, cfg.seq
)


def main() -> None:
    base = i10.I10["day"]
    icfg = replace(base, base=replace(base.base, arch=ARCH, seeds=3, note=f"day weights, {ARCH}"))
    i10.run(f"day_{ARCH}", m12.Data(), icfg)


if __name__ == "__main__":
    main()
