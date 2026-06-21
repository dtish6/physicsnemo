# Session notes — TensorBoard logs & validation metrics (2026-06-18)

Saved so I can pick this back up later. Covers: where yesterday's training log is,
how to view it, and exactly what the validation errors mean.

---

## 1. Yesterday's training log (Jun 17 run)

- **File:** `tensorboard_logs_0617_merged/events.out.tfevents.1781728255.LAPTOP-T56LBJUS.9864.0`
  (~1.9 MB, last written Jun 18 04:48)
- **Run:** 50 epochs / 71,690 steps. (Commit `c99d5fd1` message says 30 epochs, but
  the merged log actually contains 50 — run continued past the commit.)
- `eq_residual` is flat 0 → the PDE-residual loss term was off this run.

### Final scalar values

| metric                 | start → end       | min       |
|------------------------|-------------------|-----------|
| train_epoch/total      | 2.806 → 0.291     | 0.291     |
| train_epoch/mse        | 0.428 → 0.053     | 0.053     |
| train_epoch/masked_mse | 2.379 → 0.238     | 0.238     |
| val/rel_l2             | 0.382 → 0.0661    | 0.0658    |
| val/von_mises_err      | 0.434 → 0.0679    | 0.0673    |
| val/max_disp_err       | 1.67e-4 → 6.95e-5 | 6.94e-5   |
| lr                     | 1e-4 → 1.10e-6    | —         |

### View the curves (PowerShell)

```powershell
& "$env:USERPROFILE\miniconda3\envs\elasticity\python.exe" -m tensorboard.main --logdir tensorboard_logs_0617_merged
```
Then open http://localhost:6006  (add `--port 6007` if 6006 is busy).

> Note: the `elasticity` conda env python lives under **miniconda3**, not anaconda3.

---

## 2. What the `val/` errors are

Computed in `step4_predict/_4_1_metrics.py`, logged per-epoch by `validate()` in
`step3_training/_3_5_trainer.py:203`.

**Flow per epoch:** loop val set → model predicts displacement `u=(ux,uy,uz)` →
**de-normalize** pred & target to physical units → `compute_all_metrics(pred_phys, y_phys, E)`
→ average each metric over all val batches → write `val/<key>` (`_3_5_trainer.py:240`).
All tensors NCDHW = `(B,3,D,H,W)`; ν fixed at 0.3.

### `val/rel_l2` — `relative_l2_error` (`_4_1_metrics.py:22`)
Relative L2 error of the **displacement field**:
`‖u_pred − u_true‖₂ / ‖u_true‖₂`, over all voxels & 3 components, mean over batch.
Headline accuracy (~6.6% end of run). Dimensionless.

### `val/von_mises_err` — `von_mises_stress_error` (`_4_1_metrics.py:141`)
Relative L2 on the **von Mises stress field** (~6.8% end of run). Stress is derived from
displacement: central finite differences (`_fd_gradient`) → strain → isotropic Hooke's
law (λ, μ from E, ν) → Cauchy stress → von Mises scalar. Sensitive to gradient errors.

### `val/max_disp_err` — `max_displacement_error` (`_4_1_metrics.py:49`)
Max absolute error of displacement **magnitude** `‖u‖₂`:
`max_voxels |‖u_pred‖ − ‖u_true‖|` per sample, mean over batch. L∞-style worst-voxel
check, in metres (~6.9e-5 m end of run). Catches localized blow-ups L2 hides.

---

## 3. "Flatten over voxels" / "mean over batch" / batch size

- **Batch size:** `training.batch_size` in `conf/config.yaml:70` — currently **2**.
  Val loader uses the same value (`_3_5_trainer.py:310`). (config.yaml is modified in the
  working tree, so the Jun 17 run may have used a different value.)
- **Flatten over all voxels:** `reshape(B, -1)` collapses each sample's `(3,D,H,W)` into one
  long vector (for 128×128×48: `3·48·128·128 ≈ 2.36M` values), then `.norm(dim=1)` → one
  scalar per sample, shape `(B,)`. The whole field is treated as a single vector; no spatial
  structure kept.
- **Mean over batch:** `.mean()` averages those `B` per-sample numbers → 1 scalar per batch
  (`_4_1_metrics.py:212`). Trainer then accumulates over all val batches and divides by
  `n_batches` (`_3_5_trainer.py:238`) → effectively mean over every val sample (last partial
  batch weighted equally — minor bias). With batch_size=2 and a small val set, each epoch
  point averages few samples → expect some noise.

### Caveat in the code
`_3_5_trainer.py:232`: `E = x[:, 1:2]` is passed to the stress metric **still normalized**
(comment: "pattern is sufficient"). So `von_mises_err` uses a normalized modulus, not physical
Pa — fine as a *relative* trend, but absolute stress magnitudes inside aren't physical.
The displacement metrics (`rel_l2`, `max_disp_err`) ARE de-normalized / physical.
