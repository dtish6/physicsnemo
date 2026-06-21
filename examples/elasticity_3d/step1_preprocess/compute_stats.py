#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute per-channel normalization statistics for a generated dataset.

Reads the ``train/*.h5`` samples produced by ``generate_from_csv.py`` and writes
``normalization_stats.npz`` (per-channel mean/std used to z-score inputs and
targets during training). Run this AFTER generation and BEFORE ``train.py``.

This module is BOTH:
  * a standalone CLI ......... python step1_preprocess/compute_stats.py
  * a reusable function ...... compute_stats(data_dir)  (called by
                               generate_from_csv.py with ITS output folder, so
                               the stats always cover the data just written)

Pipeline position::

    1. python step1_preprocess/generate_from_csv.py   # CSVs  -> train/ val/ *.h5
    2. python step1_preprocess/compute_stats.py        # train -> normalization_stats.npz   <- THIS
    3. python train.py                                 # reads .h5 + stats, trains

Usage
-----
    # Use the DATA_DIR default below
    python step1_preprocess/compute_stats.py

    # Point at a specific dataset folder (the one with train/ inside)
    python step1_preprocess/compute_stats.py --data_dir "D:/.../hdf5_data/061611"

Stats are computed over the TRAIN split only (val must not leak into them).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable so we can reuse the training-stage dataset.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ===========================================================================
# USER CONFIG  --  used only when this script is run ON ITS OWN.
# (When generate_from_csv.py calls compute_stats(), it passes its own output
#  folder instead, so this default is ignored in that path.)
# ===========================================================================
DATA_DIR = r"D:\Nemo\physicsnemo\examples\elasticity_3d\step1_preprocess\hdf5_data\0617_within128m_3585"
# ===========================================================================


def compute_stats(data_dir, out_path=None) -> tuple[Path, dict, int]:
    """Compute per-channel normalization stats over ``<data_dir>/train``.

    Parameters
    ----------
    data_dir : str | Path
        Dataset root that contains a ``train/`` subfolder of ``*.h5`` samples.
        generate_from_csv.py passes its ``--out_dir`` here, so the stats always
        cover exactly the data that was just written.
    out_path : str | Path | None
        Destination ``.npz``. Defaults to ``<data_dir>/normalization_stats.npz``.

    Returns
    -------
    (out_path, stats_dict, n_train_files)

    Raises
    ------
    FileNotFoundError
        If ``<data_dir>/train`` is missing or has no ``*.h5`` files.
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"No train/ folder in {data_dir}. Run generate_from_csv.py first.")
    h5s = sorted(train_dir.glob("*.h5"))
    if not h5s:
        raise FileNotFoundError(f"No .h5 samples found in {train_dir}.")

    out_path = Path(out_path) if out_path else data_dir / "normalization_stats.npz"

    # Imported here (not at module top) so the heavy torch import only loads when
    # stats are actually computed -- importing this module stays cheap.
    from step3_training._3_1_dataset import ElasticityDataset  # noqa: E402
    stats = ElasticityDataset.compute_statistics(str(train_dir), str(out_path))
    return out_path, stats, len(h5s)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=DATA_DIR,
                    help="Dataset root containing a train/ subfolder. "
                         "[default: USER CONFIG DATA_DIR]")
    ap.add_argument("--out", default=None,
                    help="Output .npz path (default: <data_dir>/normalization_stats.npz).")
    args = ap.parse_args()

    train_dir = Path(args.data_dir) / "train"
    print(f"Computing normalization stats over train samples in {train_dir} ...",
          flush=True)
    out_path, stats, n = compute_stats(args.data_dir, args.out)

    print(f"\nWrote -> {out_path}  (over {n} train samples)")
    for key, arr in stats.items():
        vals = ", ".join(f"{v:.3e}" for v in arr.tolist())
        print(f"  {key:<12} [{vals}]")


if __name__ == "__main__":
    main()
