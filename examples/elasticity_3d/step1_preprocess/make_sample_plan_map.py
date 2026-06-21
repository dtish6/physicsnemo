#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the ``sample_NNNNN.h5 -> source plan_*.csv`` mapping for a
generate_from_csv run.

The train/val split is fully deterministic (seed + sorted plan list + val_frac),
so the assignment can be replayed WITHOUT the original run process. Each intended
sample is cross-checked against the files actually on disk: a sample whose .h5 is
missing was DROPPED (the FEM solve did not converge).

Two ways to use it:

  * As a helper -- generate_from_csv.py calls ``write_sample_plan_map(...)`` at
    the end of a run, passing the split it already computed in memory.

  * Standalone (e.g. for an OLD run done before this was wired in)::

        python step1_preprocess/make_sample_plan_map.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def reconstruct_split(csv_dir: str, seed: int, val_frac: float,
                      n_samples: int | None = None) -> tuple[list[str], set[int]]:
    """Replay generate_from_csv's deterministic split -> (files, val_idx).

    ``files`` is the sorted plan list; ``val_idx`` is the set of positions in
    that list assigned to the validation split. Mirrors generate_from_csv.main()
    exactly, so it reproduces the assignment of any past run with the same args.
    """
    # Local import keeps this module importable without circular-import cost
    # until the function is actually used.
    from step1_preprocess.generate_from_csv import list_field_csvs

    files = list_field_csvs(csv_dir)
    if not files:
        raise SystemExit(f"No plan_*.csv files found in {csv_dir}")
    if n_samples is not None:
        files = files[:n_samples]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    n_val = max(1, int(round(len(files) * val_frac))) if len(files) > 1 else 0
    val_idx = set(order[:n_val].tolist())
    return files, val_idx


def build_rows(files: list[str], val_idx: set[int],
               start_train: int, start_val: int,
               out_dir: Path) -> list[tuple[str, str, bool, str]]:
    """Assign each plan its sample name (split + offset) and check it on disk.

    Returns rows of ``(split, sample_name, on_disk, source_plan)`` in plan order.
    The split/offset logic is identical to generate_from_csv.main()'s task loop.
    """
    train_dir, val_dir = out_dir / "train", out_dir / "val"
    rows: list[tuple[str, str, bool, str]] = []
    n_train = n_val_w = 0
    for i, fp in enumerate(files):
        plan = Path(fp).name
        if i in val_idx:
            name = f"sample_{start_val + n_val_w:05d}.h5"; n_val_w += 1
            split, exists = "val", (val_dir / name).exists()
        else:
            name = f"sample_{start_train + n_train:05d}.h5"; n_train += 1
            split, exists = "train", (train_dir / name).exists()
        rows.append((split, name, exists, plan))
    return rows


def write_sample_plan_map(out_dir: str | Path, files: list[str], val_idx: set[int],
                          start_train: int, start_val: int) -> dict:
    """Write ``<out_dir>/sample_plan_map.csv`` and return a summary dict.

    Pass the SAME ``files`` and ``val_idx`` the run used (generate_from_csv has
    them in scope), so the map reflects the run exactly. Summary keys:
    ``path, n_assigned, n_train, n_val, n_on_disk, n_missing``.
    """
    out_dir = Path(out_dir)
    rows = build_rows(files, val_idx, start_train, start_val, out_dir)

    n_train = sum(1 for r in rows if r[0] == "train")
    n_val = len(rows) - n_train
    n_missing = sum(1 for r in rows if not r[2])

    map_path = out_dir / "sample_plan_map.csv"
    with open(map_path, "w", encoding="utf-8") as f:
        f.write("split,sample,on_disk,source_plan\n")
        for split, name, exists, plan in rows:
            f.write(f"{split},{name},{'yes' if exists else 'NO(dropped)'},{plan}\n")

    return {
        "path": map_path,
        "n_assigned": len(rows),
        "n_train": n_train,
        "n_val": n_val,
        "n_on_disk": len(rows) - n_missing,
        "n_missing": n_missing,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv_dir",
                    default=r"D:\Summer2026_ResPlan\ResPlan\fields_csv\3D_field\within_12p8m_rest2585")
    ap.add_argument("--out_dir",
                    default=r"D:\Nemo\physicsnemo\examples\elasticity_3d\step1_preprocess\hdf5_data\061711")
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start_train", type=int, default=800)
    ap.add_argument("--start_val", type=int, default=200)
    args = ap.parse_args()

    # Standalone path: reconstruct the split from the run's args, then write.
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    files, val_idx = reconstruct_split(args.csv_dir, args.seed, args.val_frac,
                                       args.n_samples)
    s = write_sample_plan_map(args.out_dir, files, val_idx,
                              args.start_train, args.start_val)

    print(f"Wrote {s['path']}")
    print(f"  Plans assigned : {s['n_assigned']}  (train={s['n_train']}, val={s['n_val']})")
    print(f"  On disk        : {s['n_on_disk']}")
    print(f"  Dropped/missing: {s['n_missing']}  (no .h5 -> FEM did not converge)")
    if s["n_missing"]:
        print("  Dropped samples (intended index has NO file on disk):")
        for split, name, exists, plan in s["rows"]:
            if not exists:
                print(f"    {split}/{name}  <-  {plan}")


if __name__ == "__main__":
    main()
