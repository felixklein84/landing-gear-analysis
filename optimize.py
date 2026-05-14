"""
optimize.py — Parametric design sweep and minimum-mass optimisation.

Strategy
--------
A uniform grid over the three free design variables (R, α, l_S) is evaluated.
For each point:
  1. Structural response is computed (model.py) with a unit load to obtain k_v.
  2. The 1-DOF impact model (impact.py) converts k_v into the actual design load
     F_impact, closing the design loop:

         geometry → k_v → F_impact → σ_max → feasibility check

  3. Feasibility is checked against all geometric, stress, and stroke constraints.
  4. Among feasible points the one with minimum leg mass is returned.

Because the structural model is linear, σ_max scales with F:
    σ_max(F) = σ_max(1 N) · F

This allows computing the unit-load response once and scaling, making the
inner loop fast without re-running the full Castigliano integration.

Optimisation objective: minimise leg mass  m_leg = ρ · A · L_path
Subject to:
    σ_max(F_impact) ≤ σ_allow          (structural stress)
    δ_max_impact    ≤ DELTA_MAX_M      (stroke / ground clearance)
    H_BOW_RANGE[0]  ≤ h_bow  ≤ H_BOW_RANGE[1]
    W_TOTAL_RANGE[0] ≤ W_total ≤ W_TOTAL_RANGE[1]
"""

import numpy as np

from model  import response, cross_section
from impact import landing_impact
import params as p


def run_sweep(n_R: int = 25, n_alpha: int = 25, n_lS: int = 15) -> dict:
    """
    Full parametric sweep over (R, α, l_S).

    Parameters
    ----------
    n_R, n_alpha, n_lS : grid resolution along each axis

    Returns
    -------
    dict with keys:
        best        : dict describing the minimum-mass feasible design point
        all_points  : list of all evaluated design point dicts
        feasible    : list of feasible design point dicts
        grids       : dict of the R_grid, alpha_grid, lS_grid arrays used
        n_total     : total number of points evaluated
    """
    cs = cross_section(p.D_OUTER, p.T_WALL)
    sigma_allow = p.MATERIAL.sigma_u / p.N_SAFETY

    R_grid     = np.linspace(*p.R_RANGE,     n_R)
    alpha_grid = np.linspace(*p.ALPHA_RANGE, n_alpha)
    lS_grid    = np.linspace(*p.LS_RANGE,    n_lS)

    all_points = []
    feasible   = []

    for R in R_grid:
        for alpha in alpha_grid:
            for l_S in lS_grid:

                # ── Unit-load structural response ──────────────────────────
                r_unit = response(
                    R, alpha, l_S, p.L_XB,
                    F=1.0, mat=p.MATERIAL, cs=cs,
                    tie_mode=p.TIE_MODE, E_tie=p.E_TIE, A_tie=p.A_TIE,
                )
                k_v = r_unit['k_v']

                # ── Impact model: derive actual design load ────────────────
                imp = landing_impact(p.M_UAV_KG, p.V_SINK_MS, k_v,
                                     n_legs=p.N_LEGS, g=p.G_MS2)
                F_design = imp['F_impact']

                # ── Scale unit-load results to actual load ─────────────────
                sigma_max = r_unit['sigma_max'] * F_design
                delta_imp = imp['delta_max']

                # ── Feasibility ────────────────────────────────────────────
                h_ok      = p.H_BOW_RANGE[0]   <= r_unit['h_bow']   <= p.H_BOW_RANGE[1]
                W_ok      = p.W_TOTAL_RANGE[0]  <= r_unit['W_total'] <= p.W_TOTAL_RANGE[1]
                sigma_ok  = sigma_max <= sigma_allow
                stroke_ok = delta_imp <= p.DELTA_MAX_M
                ok        = h_ok and W_ok and sigma_ok and stroke_ok

                pt = dict(
                    R=R, alpha=alpha, l_S=l_S,
                    h_bow=r_unit['h_bow'],
                    W_total=r_unit['W_total'],
                    k_v=k_v,
                    F_design=F_design,
                    n_dyn=imp['n_dyn'],
                    sigma_max=sigma_max,
                    delta_impact=delta_imp,
                    delta_static=imp['delta_static'],
                    H_tie=r_unit['H'] * F_design,
                    m_leg=r_unit['m_leg'],
                    feasible=ok,
                )
                all_points.append(pt)
                if ok:
                    feasible.append(pt)

    # ── Select best design ─────────────────────────────────────────────────────
    if feasible:
        best = min(feasible, key=lambda p_: p_['m_leg'])
    else:
        # No fully feasible point — report the lightest point that satisfies
        # only geometric constraints (stress/stroke may be violated).
        geom_ok = [pt for pt in all_points
                   if p.H_BOW_RANGE[0]  <= pt['h_bow']   <= p.H_BOW_RANGE[1]
                   and p.W_TOTAL_RANGE[0] <= pt['W_total'] <= p.W_TOTAL_RANGE[1]]
        candidates = geom_ok if geom_ok else all_points
        best = min(candidates, key=lambda p_: p_['sigma_max'])

    return dict(
        best=best,
        all_points=all_points,
        feasible=feasible,
        grids=dict(R=R_grid, alpha=alpha_grid, lS=lS_grid),
        n_total=len(all_points),
    )


def sigma_grid(R_grid: np.ndarray, alpha_grid: np.ndarray,
               l_S_fixed: float) -> tuple:
    """
    Compute σ_max on an (R, α) grid at fixed l_S (for contour plots).

    Returns (A_deg, R_mm, sigma_MPa, feasible_mask) as 2-D arrays.
    """
    cs          = cross_section(p.D_OUTER, p.T_WALL)
    sigma_allow = p.MATERIAL.sigma_u / p.N_SAFETY
    nR, nA      = len(R_grid), len(alpha_grid)

    A_deg, R_mm    = np.meshgrid(np.rad2deg(alpha_grid), R_grid * 1e3)
    sigma_field    = np.zeros((nR, nA))
    feasible_mask  = np.zeros((nR, nA), dtype=bool)

    for i, R in enumerate(R_grid):
        for j, alpha in enumerate(alpha_grid):
            r_unit = response(R, alpha, l_S_fixed, p.L_XB,
                              F=1.0, mat=p.MATERIAL, cs=cs,
                              tie_mode=p.TIE_MODE)
            imp = landing_impact(p.M_UAV_KG, p.V_SINK_MS, r_unit['k_v'],
                                 n_legs=p.N_LEGS, g=p.G_MS2)
            sigma = r_unit['sigma_max'] * imp['F_impact']
            sigma_field[i, j] = sigma * 1e-6

            h_ok  = p.H_BOW_RANGE[0]  <= r_unit['h_bow']   <= p.H_BOW_RANGE[1]
            W_ok  = p.W_TOTAL_RANGE[0] <= r_unit['W_total'] <= p.W_TOTAL_RANGE[1]
            feasible_mask[i, j] = (h_ok and W_ok
                                   and sigma <= sigma_allow
                                   and imp['delta_max'] <= p.DELTA_MAX_M)

    return A_deg, R_mm, sigma_field, feasible_mask
