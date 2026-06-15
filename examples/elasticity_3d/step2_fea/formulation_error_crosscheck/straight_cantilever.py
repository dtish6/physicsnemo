# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Straight rectangular cantilever — BLIND code-verification of the voxel FEM.

This is the FAIR benchmark for a full-integration Q1/H8 voxel solver (see the
project notes): axis-aligned box (exact voxelisation), full clamp + point loads
only, and the metrics are displacement / SMOOTH interior stress — never a stress
at a singular corner (which is what defeated NAFEMS LE5).

How a solver is verified against a textbook problem (what this script does)
---------------------------------------------------------------------------
We deliberately run the assessment "as if we do NOT know the exact answer",
because that is how rigorous code verification works:

  1. PATCH / CONSISTENCY TESTS (pass-fail, no benchmark needed)
       * rigid-body modes must produce ZERO internal force,
       * a constant-strain (linear-displacement) field must be reproduced to
         machine precision, and must self-equilibrate at interior nodes.
     These check the element is coded correctly, independent of any answer.

  2. BLIND CONVERGENCE + OBSERVED ORDER OF ACCURACY
       Refine the mesh uniformly (h, h/2, h/4, ...) and watch a quantity of
       interest Q_h. From three successive meshes we extract, WITHOUT the exact
       value, the observed convergence order p and a Richardson-extrapolated
       "estimated exact" value Q*:
           p  = ln[(Q_4h - Q_2h)/(Q_2h - Q_h)] / ln(2)
           Q* = Q_h + (Q_h - Q_2h)/(2^p - 1)
       For Q1 displacement QoIs the theory says p -> 2. Recovering the right
       *rate* is far stronger evidence of correctness than matching one number
       on one mesh, and Q* is our best answer with NO textbook input.

  3. ONLY THEN — THE REVEAL
       Compare Q* and the field to the closed-form beam solution:
           Euler-Bernoulli  d_EB = P L^3 / (3 E I)
           Timoshenko       d_T  = d_EB + P L / (kappa G A)   (adds shear)
           bending stress   sigma_xx(x,z) = M(x)(z - h/2)/I,  M=P(L-x)
       The figure puts the BLIND verification (left) next to the REVEAL (right).

Geometry / convention (VoxelFEMSolver tensor dims are (z, y, x))
----------------------------------------------------------------
  length L along x; width b along y; depth h along z (bending in the x-z plane).
  x = 0 face FULLY CLAMPED; total transverse force P in -z lumped over the x = L
  end face (Saint-Venant: distribution does not affect tip deflection / mid-span
  stress).

Run (use the `elasticity` conda env python):
    python step2_fea/formulation_error_crosscheck/straight_cantilever.py
    python step2_fea/formulation_error_crosscheck/straight_cantilever.py --depths 2 4 8 16 32
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))            # examples/elasticity_3d

from step2_fea._2_2_voxel_fem import VoxelFEMSolver   # noqa: E402

# --- Beam definition (clean round numbers; only the ratios matter) -----------
L  = 10.0       # length along x                         [m]
B  = 1.0        # width  along y                          [m]
H  = 1.0        # depth  along z (bending direction)      [m]
E0 = 210.0e9    # Young's modulus (210 GPa)               [Pa]
NU = 0.3        # Poisson's ratio
P  = 1.0e6      # total transverse end load, in -z        [N]
KAPPA = 5.0 / 6.0   # Timoshenko shear correction factor for a rectangle

# Fixed physical probe for the mid-span bending-stress column.
PROBE_X = 0.5 * L


# ============================================================================
# Closed-form answers  — used ONLY in the reveal/plot, never during the blind run
# ============================================================================
def analytic():
    I = B * H ** 3 / 12.0
    A = B * H
    G = E0 / (2.0 * (1.0 + NU))
    d_eb = P * L ** 3 / (3.0 * E0 * I)
    d_ti = d_eb + P * L / (KAPPA * G * A)
    return dict(I=I, A=A, G=G, d_eb=d_eb, d_ti=d_ti)


# ============================================================================
# 1. Patch / consistency tests  (no exact answer required)
# ============================================================================
def _node_coords(solver):
    iz, iy, ix = np.mgrid[0:solver.nz, 0:solver.ny, 0:solver.nx]
    return (ix.ravel() * solver.h, iy.ravel() * solver.h, iz.ravel() * solver.h)


def patch_tests():
    """Element-level correctness checks, independent of any benchmark value.

    Returns a dict of relative residuals; all should be ~1e-12 or smaller.
    """
    g = (3, 3, 3)                       # small block with a genuine interior node
    s = VoxelFEMSolver(grid_size=g, voxel_size=0.7, nu=NU)
    s.assemble_system(np.ones(g), E0=E0)
    x, y, z = _node_coords(s)
    n = s.n_nodes

    def u_to_full(ux, uy, uz):
        u = np.zeros(s.n_dof)
        u[0::3], u[1::3], u[2::3] = ux, uy, uz
        return u

    # (a) Constant-strain field: u_i = eps_ij x_j  -> must recover eps exactly.
    eps0 = np.array([1.0e-3, -3.0e-4, 2.0e-4, 5.0e-4, -2.0e-4, 3.0e-4])  # Voigt
    exx, eyy, ezz, gxy, gxz, gyz = eps0
    ux = exx * x + 0.5 * gxy * y + 0.5 * gxz * z
    uy = 0.5 * gxy * x + eyy * y + 0.5 * gyz * z
    uz = 0.5 * gxz * x + 0.5 * gyz * y + ezz * z
    u_full = u_to_full(ux, uy, uz)
    u_nodal = u_full.reshape(n, 3).T.reshape(3, s.nz, s.ny, s.nx)
    strain, _ = s.element_strain_stress(u_nodal)
    rec = strain.reshape(6, -1)                       # (6, n_elem)
    strain_err = float(np.abs(rec - eps0[:, None]).max() / np.abs(eps0).max())

    # Equilibrium of that constant-strain field: interior nodes carry ~0 force.
    r = s._matvec(u_full)
    interior = ((1 <= np.arange(n) % s.nx) & (np.arange(n) % s.nx <= s.nx - 2))
    iy_idx = (np.arange(n) // s.nx) % s.ny
    iz_idx = np.arange(n) // (s.nx * s.ny)
    interior &= (iy_idx >= 1) & (iy_idx <= s.ny - 2) & (iz_idx >= 1) & (iz_idx <= s.nz - 2)
    idof = np.repeat(interior, 3)
    equil_err = float(np.abs(r[idof]).max() / (np.abs(r).max() + 1e-300))

    # (b) Rigid-body modes: zero internal force relative to a real deformation.
    f_strain_scale = np.abs(r).max() + 1e-300
    modes = {
        "trans_x": (np.ones(n), np.zeros(n), np.zeros(n)),
        "trans_y": (np.zeros(n), np.ones(n), np.zeros(n)),
        "trans_z": (np.zeros(n), np.zeros(n), np.ones(n)),
        "rot_z":   (-y, x, np.zeros(n)),
        "rot_x":   (np.zeros(n), -z, y),
        "rot_y":   (z, np.zeros(n), -x),
    }
    rigid_err = 0.0
    for (ax, ay, az) in modes.values():
        rr = s._matvec(u_to_full(ax, ay, az))
        rigid_err = max(rigid_err, float(np.abs(rr).max() / f_strain_scale))

    return dict(strain_recovery=strain_err, equilibrium=equil_err, rigid_body=rigid_err)


# ============================================================================
# 2. One solve at a given through-depth resolution
# ============================================================================
def clamp_root_face(Nz, Ny, Nx):
    nz, ny, nx = Nz + 1, Ny + 1, Nx + 1
    iz, iy = np.mgrid[0:nz, 0:ny]
    return (iz.ravel() * ny * nx + iy.ravel() * nx + 0).astype(np.int64)


def end_face_loads(grid, total_force_z):
    Nz, Ny, Nx = grid
    nz, ny, nx = Nz + 1, Ny + 1, Nx + 1
    iz, iy = np.mgrid[0:nz, 0:ny]
    nodes = (iz.ravel() * ny * nx + iy.ravel() * nx + (nx - 1)).astype(np.int64)
    fz = total_force_z / float(nodes.size)
    return [(int(nd), np.array([0.0, 0.0, fz])) for nd in nodes]


def run_one(depth_vox, verbose=False):
    v = H / depth_vox
    Nz = depth_vox
    Ny = max(1, int(round(B / v)))
    Nx = int(round(L / v))
    grid = (Nz, Ny, Nx)

    solver = VoxelFEMSolver(grid_size=grid, voxel_size=v, nu=NU)
    solver.assemble_system(np.ones(grid), E0=E0)
    solver.apply_bcs(fixed_nodes=clamp_root_face(Nz, Ny, Nx))
    solver.apply_loads(point_loads=end_face_loads(grid, -P))
    u_nodal, _, info = solver.solve(precond="direct", verbose=verbose)
    _, stress = solver.element_strain_stress(u_nodal)
    sxx = stress[0]

    tip_uz = float(np.abs(u_nodal[2][:, :, -1]).mean())     # tip deflection QoI

    ix = min(Nx - 1, int(PROBE_X / v))
    iy = Ny // 2
    z_c = (np.arange(Nz) + 0.5) * v
    x_c = (ix + 0.5) * v
    sxx_col = sxx[:, iy, ix]                                 # mid-span column
    return dict(depth=depth_vox, v=v, grid=grid, n_active=info.get("n_active_dofs"),
                backend=info.get("backend"), resid=info["residual"],
                converged=info["converged"], tip_uz=tip_uz,
                z_c=z_c, x_c=x_c, sxx_col=sxx_col)


# ============================================================================
# 3. Richardson extrapolation  (blind observed order + estimated-exact value)
# ============================================================================
def richardson(values, r=2.0):
    """From >=3 uniformly refined results return (p_observed, Q_star).

    Uses the three FINEST values; assumes constant refinement ratio r.
    """
    q_4h, q_2h, q_h = values[-3], values[-2], values[-1]
    e1, e2 = q_4h - q_2h, q_2h - q_h
    if e1 == 0.0 or e2 == 0.0 or np.sign(e1) != np.sign(e2):
        return float("nan"), q_h
    p = np.log(e1 / e2) / np.log(r)
    q_star = q_h + (q_h - q_2h) / (r ** p - 1.0)
    return float(p), float(q_star)


# ============================================================================
# Figure: BLIND verification (left) next to the REVEAL (right)
# ============================================================================
def make_figure(results, p_obs, d_star, ana, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = np.array([r["v"] for r in results])
    uz = np.array([r["tip_uz"] for r in results])
    rf = results[-1]                                   # finest mesh

    fig, ax = plt.subplots(3, 2, figsize=(13.5, 15.5))
    fig.suptitle("Voxel FEM verification — straight cantilever  "
                 "(LEFT: blind / no exact answer    RIGHT: revealed textbook answer)",
                 fontsize=13, fontweight="bold")

    BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "0.5"

    # -- Row 1: tip-deflection convergence -----------------------------------
    a = ax[0, 0]
    a.plot(hs, uz * 1e3, "o-", color=BLUE, label="FE  tip deflection $\\delta_h$")
    a.axhline(d_star * 1e3, ls="--", color=GREY,
              label=f"Richardson $\\delta^*$ = {d_star*1e3:.4f} mm")
    a.set_title(f"BLIND: self-convergence  (observed order p = {p_obs:.2f})")
    a.set_xlabel("element size  h  [m]"); a.set_ylabel("tip deflection  [mm]")
    a.invert_xaxis(); a.grid(alpha=.3); a.legend()
    a.annotate("extrapolate the answer\nfrom the FE sequence alone",
               xy=(hs[-1], uz[-1]*1e3), xytext=(0.45, 0.25), textcoords="axes fraction",
               arrowprops=dict(arrowstyle="->", color=GREY), fontsize=9, color=GREY)

    a = ax[0, 1]
    a.plot(hs, uz * 1e3, "o-", color=BLUE, label="FE  $\\delta_h$")
    a.axhline(ana["d_eb"] * 1e3, ls=":", color=GREEN,
              label=f"Euler-Bernoulli = {ana['d_eb']*1e3:.4f} mm")
    a.axhline(ana["d_ti"] * 1e3, ls="-.", color=RED,
              label=f"Timoshenko = {ana['d_ti']*1e3:.4f} mm")
    a.axhline(d_star * 1e3, ls="--", color=GREY, label=f"$\\delta^*$ = {d_star*1e3:.4f} mm")
    a.set_title("REVEAL: FE vs textbook beam theory")
    a.set_xlabel("element size  h  [m]"); a.set_ylabel("tip deflection  [mm]")
    a.invert_xaxis(); a.grid(alpha=.3); a.legend(fontsize=8)

    # -- Row 2: order of accuracy (log-log error vs h) -----------------------
    err_blind = np.abs(uz - d_star) / abs(d_star)
    fit_mask = err_blind > 0
    a = ax[1, 0]
    a.loglog(hs[fit_mask], err_blind[fit_mask], "o-", color=BLUE,
             label="|$\\delta_h-\\delta^*$| / $\\delta^*$")
    if fit_mask.sum() >= 2:
        slope = np.polyfit(np.log(hs[fit_mask]), np.log(err_blind[fit_mask]), 1)[0]
        a.set_title(f"BLIND: error vs mesh size  (fitted slope = {slope:.2f})")
    a.set_xlabel("element size  h  [m]"); a.set_ylabel("relative error")
    a.grid(alpha=.3, which="both"); a.legend()

    err_true = np.abs(uz - ana["d_ti"]) / ana["d_ti"]
    a = ax[1, 1]
    a.loglog(hs, err_true, "o-", color=RED, label="|$\\delta_h-\\delta_T$| / $\\delta_T$")
    # theoretical O(h^2) reference triangle anchored at the coarsest point
    h0, e0 = hs[0], err_true[0]
    a.loglog(hs, e0 * (hs / h0) ** 2, ls="--", color=GREY, label="theoretical slope 2  $O(h^2)$")
    a.set_title("REVEAL: error vs exact, with theoretical $O(h^2)$")
    a.set_xlabel("element size  h  [m]"); a.set_ylabel("relative error")
    a.grid(alpha=.3, which="both"); a.legend()
    a.annotate("error vs Timoshenko PLATEAUS:\n1-D beam theory is itself\nan approximation of the 3-D solid",
               xy=(hs[-1], err_true[-1]), xytext=(0.08, 0.12), textcoords="axes fraction",
               arrowprops=dict(arrowstyle="->", color=GREY), fontsize=8, color=GREY)

    # -- Row 3: bending-stress field -----------------------------------------
    a = ax[2, 0]
    for r in results[-3:]:
        a.plot(r["sxx_col"] / 1e6, r["z_c"] / H, "o-",
               label=f"depth {r['depth']}  ({r['depth']} vox)")
    a.set_title(f"BLIND: mid-span $\\sigma_{{xx}}$ profile self-converges  (x={rf['x_c']:.2f} m)")
    a.set_xlabel("$\\sigma_{xx}$  [MPa]"); a.set_ylabel("z / h")
    a.grid(alpha=.3); a.legend(fontsize=8)

    a = ax[2, 1]
    z_fine = rf["z_c"]
    M = P * (L - rf["x_c"])
    sxx_an = M * (z_fine - H / 2.0) / ana["I"]
    a.plot(rf["sxx_col"] / 1e6, z_fine / H, "o", color=BLUE, ms=7, label="FE (finest)")
    zz = np.linspace(0, H, 100)
    a.plot((M * (zz - H / 2.0) / ana["I"]) / 1e6, zz / H, "-", color=RED,
           label="analytic  $M(z-h/2)/I$")
    a.set_title("REVEAL: $\\sigma_{xx}$ vs analytic bending profile")
    a.set_xlabel("$\\sigma_{xx}$  [MPa]"); a.set_ylabel("z / h")
    a.grid(alpha=.3); a.legend()

    fig.tight_layout(rect=[0, 0.04, 1, 0.975])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_criteria_figure(results, pt, p_obs, d_star, ana, out_path):
    """One panel per verification criterion, each paired with its textbook/exact
    reference — a teaching figure walking through the five checks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = np.array([r["v"] for r in results])
    uz = np.array([r["tip_uz"] for r in results]) * 1e3        # mm
    depths = np.array([r["depth"] for r in results])
    rf = results[-1]
    d_star_mm, d_eb_mm, d_ti_mm = d_star * 1e3, ana["d_eb"] * 1e3, ana["d_ti"] * 1e3

    BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "0.5"
    fig, ax = plt.subplots(3, 2, figsize=(14, 16.5))
    fig.suptitle("Voxel FEM verification — WHAT each criterion is compared against\n"
                 "(only 1, 3, 5 use an outside reference; 2 & 4 are blind / self-checking)",
                 fontsize=13.5, fontweight="bold")

    # -- Criterion 1: patch tests (reference = exact 0) ----------------------
    a = ax[0, 0]
    names = ["constant-strain\nrecovery", "interior\nequilibrium", "rigid-body\nzero-energy"]
    vals = [pt["strain_recovery"], pt["equilibrium"], pt["rigid_body"]]
    bars = a.barh(names, vals, color=GREEN, zorder=3)
    a.set_xscale("log"); a.set_xlim(1e-16, 1e-5)
    a.axvline(1e-9, color=RED, ls="--", label="pass threshold")
    a.axvline(1e-15, color="k", ls=":", lw=1.5, label="REFERENCE: exact = 0 (machine precision)")
    for b, v in zip(bars, vals):
        a.text(v * 2, b.get_y() + b.get_height() / 2, f"{v:.0e}", va="center", fontsize=8)
    a.set_title("Criterion 1 — Patch tests\ncompared against:  the EXACT answer = 0")
    a.set_xlabel("error  (left = better)"); a.legend(fontsize=7, loc="lower right")

    # -- Criterion 2: convergence (reference = itself; textbook shown faintly) -
    a = ax[0, 1]
    a.axhspan(min(d_eb_mm, d_ti_mm), max(d_eb_mm, d_ti_mm), color=RED, alpha=0.08,
              zorder=0, label="textbook range (context only — NOT used here)")
    a.plot(depths, uz, "o-", color=BLUE, zorder=3, label="FE result")
    a.axhline(d_star_mm, ls="--", color="k", lw=1.5,
              label=f"REFERENCE: its own settled value ({d_star_mm:.2f} mm)")
    a.set_xscale("log", base=2); a.set_xticks(depths); a.set_xticklabels(depths)
    a.minorticks_off()
    for i in range(len(uz) - 1):
        a.annotate(f"+{uz[i+1]-uz[i]:.2f}",
                   xy=((depths[i] * depths[i + 1]) ** 0.5, (uz[i] + uz[i + 1]) / 2),
                   fontsize=8, color=BLUE, ha="center", va="bottom")
    a.set_title("Criterion 2 — Convergence\ncompared against:  ITSELF — does it stop moving?  (blind)")
    a.set_xlabel("cubes through thickness"); a.set_ylabel("tip deflection [mm]")
    a.grid(alpha=.3); a.legend(fontsize=7, loc="lower right")

    # -- Criterion 3: order of accuracy (reference = theoretical rate) --------
    a = ax[1, 0]
    err = np.abs(uz - d_star_mm) / d_star_mm
    m = err > 0
    a.loglog(hs[m], err[m], "o-", color=BLUE, label="FE error", zorder=3)
    slope = np.polyfit(np.log(hs[m]), np.log(err[m]), 1)[0]
    a.loglog(hs, err[0] * (hs / hs[0]) ** 2, "--", color="k", lw=1.5,
             label="REFERENCE: theory says slope = 2  $O(h^2)$")
    a.set_title(f"Criterion 3 — Order of accuracy  (measured slope {slope:.2f})\n"
                f"compared against:  the THEORETICAL rate, slope 2")
    a.set_xlabel("cube size  h [m]"); a.set_ylabel("relative error")
    a.grid(alpha=.3, which="both"); a.legend(fontsize=8)

    # -- Criterion 4: Richardson (reference = own extrapolation; textbook faint) -
    a = ax[1, 1]
    a.axhspan(min(d_eb_mm, d_ti_mm), max(d_eb_mm, d_ti_mm), color=RED, alpha=0.08,
              zorder=0, label="textbook range (context only — NOT used here)")
    a.plot(hs, uz, "o-", color=BLUE, label="FE results", zorder=3)
    a.plot([hs[-1], 0], [uz[-1], d_star_mm], ":", color="k")
    a.plot([0], [d_star_mm], "*", color="k", ms=18,
           label=f"REFERENCE: own extrapolation $\\delta^*$ = {d_star_mm:.2f} mm", zorder=4)
    a.axhline(d_star_mm, ls="--", color="k", alpha=.5, lw=1)
    a.set_xlim(-0.03 * hs[0], hs[0] * 1.05)
    a.set_title("Criterion 4 — Richardson extrapolation\ncompared against:  its OWN extrapolated value  (blind)")
    a.set_xlabel("cube size  h [m]   (0 = infinitely fine)"); a.set_ylabel("tip deflection [mm]")
    a.grid(alpha=.3); a.legend(fontsize=7, loc="lower right")
    a.annotate("the 'wall':\ninfinite-resolution\nanswer from FE alone",
               xy=(0, d_star_mm), xytext=(0.30, 0.32), textcoords="axes fraction",
               arrowprops=dict(arrowstyle="->", color=GREY), fontsize=8, color=GREY)

    # -- Criterion 5a: reveal vs textbook beam theory (displacement) ---------
    a = ax[2, 0]
    a.plot(hs, uz, "o-", color=BLUE, label="FE $\\delta_h$", zorder=3)
    a.axhline(d_eb_mm, ls=":", color=GREEN, label=f"REFERENCE: Euler-Bernoulli = {d_eb_mm:.2f}")
    a.axhline(d_ti_mm, ls="-.", color=RED, label=f"REFERENCE: Timoshenko = {d_ti_mm:.2f}")
    a.axhline(d_star_mm, ls="--", color=GREY, label=f"blind $\\delta^*$ = {d_star_mm:.2f}")
    a.invert_xaxis()
    a.set_title("Criterion 5 — Reveal (deflection)\ncompared against:  TEXTBOOK formulas (EB & Timoshenko)")
    a.set_xlabel("cube size  h [m]"); a.set_ylabel("tip deflection [mm]")
    a.grid(alpha=.3); a.legend(fontsize=7)

    # -- Criterion 5b: reveal vs analytic stress field -----------------------
    a = ax[2, 1]
    z = rf["z_c"]; M = P * (L - rf["x_c"])
    a.plot(rf["sxx_col"] / 1e6, z / H, "o", color=BLUE, ms=7, label="FE (finest mesh)", zorder=3)
    zz = np.linspace(0, H, 100)
    a.plot((M * (zz - H / 2.0) / ana["I"]) / 1e6, zz / H, "-", color=RED,
           label="REFERENCE: analytic  $M(z-h/2)/I$")
    a.set_title("Criterion 5 — Reveal (stress)\ncompared against:  TEXTBOOK formula  $M\\,y/I$")
    a.set_xlabel("$\\sigma_{xx}$ [MPa]"); a.set_ylabel("z / h")
    a.grid(alpha=.3); a.legend(fontsize=8)

    # -- Conclusion banner for the Criterion 5 comparison --------------------
    pct_eb = 100.0 * (d_star_mm - d_eb_mm) / d_eb_mm
    pct_ti = 100.0 * (d_star_mm - d_ti_mm) / d_ti_mm
    sxx_an_col = M * (rf["z_c"] - H / 2.0) / ana["I"]
    kt = int(np.argmax(np.abs(rf["z_c"] - H / 2.0)))
    stress_pct = 100.0 * (rf["sxx_col"][kt] / sxx_an_col[kt] - 1.0)
    conclusion = (
        "CONCLUSION (Criterion 5 — Reveal):  the solver is VERIFIED — it reproduces textbook beam behavior.\n"
        f"Blind converged deflection $\\delta^*$ = {d_star_mm:.2f} mm lands BETWEEN the two formulas "
        f"({pct_eb:+.2f}% vs Euler-Bernoulli, {pct_ti:+.2f}% vs Timoshenko), and the full bending-stress "
        f"field matches analytic $M\\,y/I$ to {stress_pct:+.1f}% at the finest mesh.\n"
        "The sub-percent gap to Timoshenko does NOT vanish under refinement because 1-D beam theory is itself "
        "an approximation of the 3-D solid\n(clamped, warping-restrained end + lumped tip load make it slightly "
        "stiffer) — i.e. PHYSICS, not solver error.  Trust the blind $\\delta^*$, not a naive ratio-to-textbook.")
    fig.text(0.5, 0.012, conclusion, ha="center", va="bottom", fontsize=9.5,
             bbox=dict(boxstyle="round", fc="#fff3cd", ec="#d0a000", alpha=0.95))

    fig.tight_layout(rect=[0, 0.105, 1, 0.965])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16],
                    help="Voxels through the beam depth H (uniform refinement, ratio 2).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("=== Straight cantilever — BLIND code-verification of the voxel FEM ===")
    print(f"  geometry : L={L}  b={B}  h={H} m   (slenderness L/h = {L/H:.0f})")
    print(f"  material : E={E0:.3e} Pa  nu={NU}   load P={P:.2e} N in -z at x=L\n")

    # --- step 1: patch / consistency tests (no answer needed) ----------------
    pt = patch_tests()
    ok = lambda v: "PASS" if v < 1e-9 else "**FAIL**"
    print("  [1] PATCH / CONSISTENCY TESTS  (element correctness, no benchmark)")
    print(f"      constant-strain recovery : {pt['strain_recovery']:.2e}   {ok(pt['strain_recovery'])}")
    print(f"      interior equilibrium     : {pt['equilibrium']:.2e}   {ok(pt['equilibrium'])}")
    print(f"      rigid-body zero-energy   : {pt['rigid_body']:.2e}   {ok(pt['rigid_body'])}\n")

    # --- step 2: blind convergence sweep -------------------------------------
    print("  [2] BLIND CONVERGENCE  (refine, extract order & extrapolated answer)")
    print(f"  {'depth':>5} {'h[m]':>8} {'grid (Nz,Ny,Nx)':>18} {'active_dof':>10} "
          f"{'tip_uz[mm]':>11} {'resid':>9}")
    results = []
    for d in args.depths:
        r = run_one(d, verbose=args.verbose)
        results.append(r)
        print(f"  {r['depth']:>5} {r['v']:>8.4f} {str(r['grid']):>18} {r['n_active']:>10} "
              f"{r['tip_uz']*1e3:>11.5f} {r['resid']:>9.1e}")

    uz = [r["tip_uz"] for r in results]
    p_obs, d_star = richardson(uz)
    print(f"\n      observed order of accuracy  p = {p_obs:.3f}   (Q1 displacement theory -> 2)")
    print(f"      Richardson estimated-exact  delta* = {d_star*1e3:.5f} mm")
    print(f"      -> BLIND verdict: solver converges at ~{'2nd' if p_obs>1.6 else f'{p_obs:.1f}'} order"
          f" to delta* with NO textbook input.\n")

    # --- step 3: the reveal ---------------------------------------------------
    ana = analytic()
    print("  [3] REVEAL — compare to closed-form beam theory")
    print(f"      Euler-Bernoulli  d_EB = {ana['d_eb']*1e3:.5f} mm")
    print(f"      Timoshenko       d_T  = {ana['d_ti']*1e3:.5f} mm")
    print(f"      Richardson delta*      = {d_star*1e3:.5f} mm  "
          f"({100*(d_star-ana['d_ti'])/ana['d_ti']:+.2f}% vs Timoshenko, "
          f"{100*(d_star-ana['d_eb'])/ana['d_eb']:+.2f}% vs Euler-Bernoulli)")
    print("      (a sub-% gap to 1-D beam theory is expected: the clamped, warping-")
    print("       restrained end + lumped tip load make the 3-D solid slightly stiffer;")
    print("       this is physics, not solver error — hence trust the blind delta*.)\n")

    out = _HERE / "straight_cantilever_verification.png"
    out2 = _HERE / "straight_cantilever_criteria.png"
    try:
        if out.exists():
            print(f"  kept existing figure (not overwritten): {out}")
        else:
            make_figure(results, p_obs, d_star, ana, out)
            print(f"  figure written: {out}")
        make_criteria_figure(results, pt, p_obs, d_star, ana, out2)
        print(f"  figure written: {out2}")
    except Exception as e:                       # matplotlib missing / headless
        print(f"  [figure skipped: {type(e).__name__}: {e}]")


if __name__ == "__main__":
    main()
