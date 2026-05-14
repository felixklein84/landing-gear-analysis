"""
run.py — Main analysis script for UAV landing gear design.

Workflow
--------
1. Print design inputs and cross-section properties.
2. Evaluate a single reference geometry (with and without tie) for comparison.
3. Run the 1-DOF impact model to derive the actual design load from UAV specs.
4. Execute the parametric sweep to find the minimum-mass feasible design.
5. Size the tie member for the optimal design.
6. Print a full design summary.
7. Generate and save four figure panels.

Usage
-----
    python run.py

Output files
------------
    fig1_geometry_and_stress.png   — gear shape + moment diagram + stress envelope
    fig2_impact_dynamics.png       — force-time history + design load derivation
    fig3_design_space.png          — σ_max contour map with feasible region
    fig4_mass_vs_stiffness.png     — Pareto view of all feasible points
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model    import response, cross_section, compute_path
from impact   import landing_impact, impact_time_history
from optimize import run_sweep, sigma_grid
import params as p


# ── Convenience ───────────────────────────────────────────────────────────────

def _cs():
    return cross_section(p.D_OUTER, p.T_WALL)


def _sigma_allow():
    return p.MATERIAL.sigma_u / p.N_SAFETY


# ── 1. Design inputs ───────────────────────────────────────────────────────────

def print_inputs(cs):
    SEP = "=" * 72
    print(SEP)
    print("  UAV LANDING GEAR  —  Curved Skid-Bow + Tied Arch")
    print("  HORYZN Aerostructures")
    print(SEP)
    print(f"  Material      : {p.MATERIAL.name}")
    print(f"                  E      = {p.MATERIAL.E*1e-9:.1f} GPa")
    print(f"                  σ_y    = {p.MATERIAL.sigma_y*1e-6:.0f} MPa")
    print(f"                  σ_u    = {p.MATERIAL.sigma_u*1e-6:.0f} MPa")
    print(f"                  ρ      = {p.MATERIAL.rho:.0f} kg/m³")
    print(f"  Cross-section : D_o = {p.D_OUTER*1e3:.1f} mm,  t = {p.T_WALL*1e3:.2f} mm")
    print(f"                  A   = {cs['A']*1e6:.1f} mm²")
    print(f"                  I   = {cs['I']*1e12:.0f} mm⁴")
    print(f"                  W_b = {cs['W_b']*1e9:.0f} mm³")
    print(f"  UAV specs     : m = {p.M_UAV_KG:.0f} kg,  v_sink = {p.V_SINK_MS:.1f} m/s,"
          f"  n_legs = {p.N_LEGS}")
    print(f"  Safety factor : n = {p.N_SAFETY:.1f}  →  σ_allow = {_sigma_allow()*1e-6:.1f} MPa")
    print(f"  Crossbeam     : L_xb = {p.L_XB*1e3:.0f} mm")
    print(f"  Tie mode      : {p.TIE_MODE!r}")
    print()


# ── 2. Reference comparison: no tie vs. rigid tie ─────────────────────────────

R_REF     = 0.403
ALPHA_REF = np.deg2rad(70.0)
LS_REF    = 0.250


def print_reference_comparison(cs):
    F_ref = p.F_STATIC_PER_LEG   # quasi-static load for comparison only
    r0 = response(R_REF, ALPHA_REF, LS_REF, p.L_XB, F_ref,
                  p.MATERIAL, cs, tie_mode='none')
    r1 = response(R_REF, ALPHA_REF, LS_REF, p.L_XB, F_ref,
                  p.MATERIAL, cs, tie_mode='rigid')

    print("  REFERENCE GEOMETRY  (R=403 mm, α=70°, l_S=250 mm)")
    print("  Quasi-static load F = {:.0f} N  (m·g / n_legs)".format(F_ref))
    print("-" * 72)
    print(f"  {'':32s}  {'No tie':>12s}    {'Rigid tie':>12s}")
    print(f"  {'Gear height h':32s}  {r0['h_bow']*1e3:>10.1f} mm  {r1['h_bow']*1e3:>12.1f} mm")
    print(f"  {'Track width W':32s}  {r0['W_total']*1e3:>10.1f} mm  {r1['W_total']*1e3:>12.1f} mm")
    print(f"  {'Tie force H':32s}  {0:>12.0f} N  {r1['H']:>14.0f} N")
    print(f"  {'Gear stroke δ':32s}  {r0['delta']*1e3:>10.2f} mm  {r1['delta']*1e3:>12.2f} mm")
    print(f"  {'Vertical stiffness k_v':32s}  {r0['k_v']*1e-3:>9.1f} kN/m  {r1['k_v']*1e-3:>11.1f} kN/m")
    print(f"  {'Peak bending stress σ_max':32s}  {r0['sigma_max']*1e-6:>9.1f} MPa  {r1['sigma_max']*1e-6:>11.1f} MPa")
    print(f"  {'  at segment':32s}  {r0['loc_max']:>12s}  {r1['loc_max']:>14s}")
    sa = _sigma_allow()
    print(f"  {'Stress margin vs σ_u/n':32s}  "
          f"{(sa/r0['sigma_max']-1)*100:>+9.1f} %  "
          f"{(sa/r1['sigma_max']-1)*100:>+11.1f} %")
    print()
    return r0, r1


# ── 3. Impact model ────────────────────────────────────────────────────────────

def print_impact(k_v_ref: float):
    imp = landing_impact(p.M_UAV_KG, p.V_SINK_MS, k_v_ref,
                         n_legs=p.N_LEGS, g=p.G_MS2)
    print("  LANDING IMPACT  (1-DOF spring-mass,  reference geometry with tie)")
    print("-" * 72)
    print(f"  Natural freq ω      = {imp['omega']:.1f} rad/s"
          f"  ({imp['omega']/(2*np.pi):.2f} Hz)")
    print(f"  Static deflection   = {imp['delta_static']*1e3:.1f} mm")
    print(f"  Peak stroke δ_max   = {imp['delta_max']*1e3:.1f} mm")
    print(f"  Peak impact force   = {imp['F_impact']:.0f} N  per leg")
    print(f"  Dynamic load factor = {imp['n_dyn']:.2f}  ×  quasi-static")
    print(f"  Kinetic energy      = {imp['E_kinetic']:.1f} J  (per leg)")
    print(f"  Elastic capacity    = {imp['E_elastic']:.1f} J  (per leg)")
    print(f"  Energy balance OK?  {imp['energy_ok']}")
    print()
    return imp


# ── 4. Optimisation ────────────────────────────────────────────────────────────

def print_optimisation(sweep: dict):
    best = sweep['best']
    sa   = _sigma_allow()
    n_feas = len(sweep['feasible'])
    print(f"  PARAMETRIC SWEEP  ({sweep['n_total']} points,  "
          f"{n_feas} feasible)")
    print("-" * 72)
    if n_feas == 0:
        print("  WARNING: no fully feasible point found — showing best available.")
    print(f"  Optimum (min mass):")
    print(f"    R        = {best['R']*1e3:.0f} mm")
    print(f"    α        = {np.rad2deg(best['alpha']):.1f}°")
    print(f"    l_S      = {best['l_S']*1e3:.0f} mm")
    print(f"    h_bow    = {best['h_bow']*1e3:.0f} mm")
    print(f"    W_total  = {best['W_total']*1e3:.0f} mm")
    print(f"    k_v      = {best['k_v']*1e-3:.1f} kN/m")
    print(f"    n_dyn    = {best['n_dyn']:.2f}")
    print(f"    F_design = {best['F_design']:.0f} N  per leg")
    print(f"    σ_max    = {best['sigma_max']*1e-6:.1f} MPa")
    print(f"    Margin vs σ_u/n  = {(sa/best['sigma_max']-1)*100:+.1f} %")
    print(f"    Margin vs σ_y    = {(p.MATERIAL.sigma_y/best['sigma_max']-1)*100:+.1f} %")
    print(f"    δ_max (impact)   = {best['delta_impact']*1e3:.1f} mm")
    print(f"    H_tie    = {best['H_tie']/1000:.2f} kN")
    print(f"    m_leg    = {best['m_leg']*1e3:.0f} g  (half-leg, foot to midpoint)")
    print()


# ── 5. Tie sizing ──────────────────────────────────────────────────────────────

def print_tie_sizing(H_design: float):
    print("  TIE MEMBER SIZING  (tension only)")
    print("-" * 72)
    print(f"  Design tension H = {H_design/1000:.2f} kN")

    candidates = [
        ("Steel cable",    750e6,  "σ_zul = σ_u/2 = 750 MPa"),
        ("Al 7075-T6 rod", p.MATERIAL.sigma_u / p.N_SAFETY,
         f"σ_zul = σ_u/n = {p.MATERIAL.sigma_u/p.N_SAFETY*1e-6:.0f} MPa"),
        ("CFRP pultruded", 1000e6, "σ_zul = 1000 MPa (conservative)"),
    ]
    for name, sigma_zul, note in candidates:
        A_min = H_design / sigma_zul
        d_min = np.sqrt(4 * A_min / np.pi)
        print(f"  {name:<20s}  ({note})")
        print(f"    A_min = {A_min*1e6:.2f} mm²  →  d_min ≈ {d_min*1e3:.2f} mm")
    print()


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _plot_gear_shape(ax, R, alpha, l_S, L_xb, H, label=''):
    path    = compute_path(R, alpha, l_S, L_xb, n=400)
    skid, arc, xb = path
    foot_x  = (L_xb + 2 * (R * np.sin(alpha) + l_S * np.cos(alpha))) / 2

    colors = {'skid': '#e05c5c', 'arc': '#4c9e4c', 'xb': '#3a7abf'}
    # Right half
    for seg, name in zip([skid, arc, xb], ['skid', 'arc', 'xb']):
        x_r = foot_x - seg['x']
        ax.plot( x_r, seg['y'], color=colors[name], lw=2.5)
        ax.plot(-x_r, seg['y'], color=colors[name], lw=2.5)

    # Crossbeam
    y_apex = xb['y'][0]
    ax.plot([-L_xb/2, L_xb/2], [y_apex, y_apex], color=colors['xb'], lw=3.0)

    # Tie
    ax.plot([-foot_x + 0.005, foot_x - 0.005], [0, 0],
            color='#7b5ea7', lw=2, ls='--', label=f'Tie  H={H/1000:.1f} kN')

    # Ground line
    ax.axhline(-0.004, color='#888', lw=0.8)

    # Segment legend patches
    patches = [
        mpatches.Patch(color=colors['skid'], label='Skid'),
        mpatches.Patch(color=colors['arc'],  label='Bow (arc)'),
        mpatches.Patch(color=colors['xb'],   label='Crossbeam'),
    ]
    return patches, foot_x


def _collect(path, key):
    s = np.concatenate([seg['s'] for seg in path])
    v = np.concatenate([seg[key] for seg in path])
    return s, v


# ── Figure 1: Geometry + moment + stress ──────────────────────────────────────

def plot_fig1(best: dict, cs: dict):
    R, alpha, l_S = best['R'], best['alpha'], best['l_S']
    F = best['F_design']
    sa = _sigma_allow()

    r_no  = response(R, alpha, l_S, p.L_XB, F, p.MATERIAL, cs, tie_mode='none')
    r_yes = response(R, alpha, l_S, p.L_XB, F, p.MATERIAL, cs, tie_mode='rigid')

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"Optimal design  —  R={R*1e3:.0f} mm, α={np.rad2deg(alpha):.0f}°, "
        f"l_S={l_S*1e3:.0f} mm\n"
        f"F_design={F/1000:.2f} kN/leg  (m={p.M_UAV_KG:.0f} kg, "
        f"v_sink={p.V_SINK_MS:.1f} m/s, n_dyn={best['n_dyn']:.2f})",
        fontsize=10,
    )

    # (a) Gear shape
    ax = axs[0]
    patches, _ = _plot_gear_shape(ax, R, alpha, l_S, p.L_XB, r_yes['H'])
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Gear geometry')
    ax.grid(alpha=0.3)
    ax.legend(handles=patches + [
        mpatches.Patch(color='#7b5ea7', label=f'Tie  H={r_yes["H"]/1000:.1f} kN')
    ], fontsize=8, loc='upper right')

    # (b) Bending moment
    ax = axs[1]
    s_n, M_n = _collect(r_no['path'],  'M')
    s_y, M_y = _collect(r_yes['path'], 'M')
    b1 = l_S
    b2 = l_S + R * alpha
    ax.plot(s_n * 1e3, M_n, color='#e05c5c', lw=2, label='No tie')
    ax.plot(s_y * 1e3, M_y, color='#4c9e4c', lw=2, label='Rigid tie')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(b1 * 1e3, color='gray',  ls=':', lw=1, label='Skid → arc')
    ax.axvline(b2 * 1e3, color='black', ls=':', lw=1, label='Arc → xbeam')
    ax.set_xlabel('Path length from foot s [mm]')
    ax.set_ylabel('M [N·m]')
    ax.set_title('Bending moment along half-leg')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (c) Bending stress
    ax = axs[2]
    s_n2, sg_n = _collect(r_no['path'],  'sigma')
    s_y2, sg_y = _collect(r_yes['path'], 'sigma')
    ax.plot(s_n2 * 1e3, sg_n * 1e-6, color='#e05c5c', lw=2, label='No tie')
    ax.plot(s_y2 * 1e3, sg_y * 1e-6, color='#4c9e4c', lw=2, label='Rigid tie')
    for val, col, ls, lbl in [
        ( sa * 1e-6,                 '#e05c5c', '--', f'σ_u/n = {sa*1e-6:.0f} MPa'),
        (-sa * 1e-6,                 '#e05c5c', '--', None),
        ( p.MATERIAL.sigma_y * 1e-6, 'orange',  '-.', f'σ_y = {p.MATERIAL.sigma_y*1e-6:.0f} MPa'),
        (-p.MATERIAL.sigma_y * 1e-6, 'orange',  '-.', None),
        ( p.MATERIAL.sigma_u * 1e-6, '#333',    ':',  f'σ_u = {p.MATERIAL.sigma_u*1e-6:.0f} MPa'),
        (-p.MATERIAL.sigma_u * 1e-6, '#333',    ':',  None),
    ]:
        ax.axhline(val, color=col, ls=ls, lw=1.2, alpha=0.7, label=lbl)
    ax.axvline(b1 * 1e3, color='gray',  ls=':', lw=1)
    ax.axvline(b2 * 1e3, color='black', ls=':', lw=1)
    ax.set_xlabel('Path length from foot s [mm]')
    ax.set_ylabel('σ [MPa]')
    ax.set_title('Bending stress along half-leg')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper left', ncol=2)

    plt.tight_layout()
    fname = 'fig1_geometry_and_stress.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ── Figure 2: Landing impact dynamics ─────────────────────────────────────────

def plot_fig2(best: dict):
    k_v = best['k_v']
    imp = landing_impact(p.M_UAV_KG, p.V_SINK_MS, k_v,
                         n_legs=p.N_LEGS, g=p.G_MS2)
    hist = impact_time_history(p.M_UAV_KG, p.V_SINK_MS, k_v,
                               n_legs=p.N_LEGS, g=p.G_MS2)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Landing impact dynamics  (1-DOF linear spring-mass, no damping)', fontsize=10)

    # (a) Contact force vs time
    ax = axs[0]
    t_ms = hist['t'] * 1e3
    F_kN = hist['F_contact'] / 1000
    ax.plot(t_ms, F_kN, color='#3a7abf', lw=2.5)
    ax.axhline(imp['F_impact'] / 1000, color='#e05c5c', ls='--', lw=1.5,
               label=f"F_peak = {imp['F_impact']/1000:.2f} kN")
    ax.axhline(imp['F_static'] / 1000, color='gray', ls=':', lw=1.2,
               label=f"F_static = {imp['F_static']/1000:.2f} kN")
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Contact force per leg [kN]')
    ax.set_title(f'v_sink = {p.V_SINK_MS:.1f} m/s,  '
                 f'k_v = {k_v/1000:.1f} kN/m,  '
                 f'n_dyn = {imp["n_dyn"]:.2f}')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # (b) Dynamic load factor vs sink rate (sensitivity)
    v_range   = np.linspace(0.5, 4.0, 60)
    n_dyn_arr = []
    for v in v_range:
        imp_v = landing_impact(p.M_UAV_KG, v, k_v,
                               n_legs=p.N_LEGS, g=p.G_MS2)
        n_dyn_arr.append(imp_v['n_dyn'])

    ax2 = axs[1]
    ax2.plot(v_range, n_dyn_arr, color='#3a7abf', lw=2.5)
    ax2.axvline(p.V_SINK_MS, color='#4c9e4c', ls='--', lw=1.5,
                label=f'Design v_sink = {p.V_SINK_MS:.1f} m/s')
    ax2.axvline(2.5, color='orange', ls=':', lw=1.2,
                label='EASA SC-VTOL normal limit (2.5 m/s)')
    ax2.axvline(3.5, color='#e05c5c', ls=':', lw=1.2,
                label='EASA SC-VTOL hard landing (3.5 m/s)')
    ax2.set_xlabel('Sink rate [m/s]')
    ax2.set_ylabel('Dynamic load factor n_dyn  [-]')
    ax2.set_title('Sensitivity: n_dyn vs sink rate')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fname = 'fig2_impact_dynamics.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ── Figure 3: Design space (σ_max contour) ────────────────────────────────────

def plot_fig3(sweep: dict):
    best     = sweep['best']
    R_grid   = sweep['grids']['R']
    a_grid   = sweep['grids']['alpha']
    l_S_opt  = best['l_S']

    A_deg, R_mm, sigma_field, feas_mask = sigma_grid(R_grid, a_grid, l_S_opt)
    sa = _sigma_allow()

    fig, ax = plt.subplots(figsize=(8, 6))
    cs_fill = ax.contourf(A_deg, R_mm, sigma_field, levels=25, cmap='viridis')
    ax.contour(A_deg, R_mm, sigma_field,
               levels=[sa * 1e-6], colors='red',
               linewidths=2, linestyles='--')
    ax.contour(A_deg, R_mm, sigma_field,
               levels=[p.MATERIAL.sigma_y * 1e-6], colors='orange',
               linewidths=1.5, linestyles='-.')
    ax.contour(A_deg, R_mm, sigma_field,
               levels=[p.MATERIAL.sigma_u * 1e-6], colors='white',
               linewidths=1, linestyles=':')
    ax.plot(np.rad2deg(best['alpha']), best['R'] * 1e3,
            'k*', markersize=18, label='Optimum (min mass)', zorder=5)
    plt.colorbar(cs_fill, ax=ax, label='σ_max [MPa]')

    legend_lines = [
        plt.Line2D([0], [0], color='red',    ls='--', lw=2,
                   label=f"σ_u/n = {sa*1e-6:.0f} MPa  (design limit)"),
        plt.Line2D([0], [0], color='orange', ls='-.', lw=1.5,
                   label=f"σ_y  = {p.MATERIAL.sigma_y*1e-6:.0f} MPa  (yield)"),
        plt.Line2D([0], [0], color='white',  ls=':',  lw=1,
                   label=f"σ_u  = {p.MATERIAL.sigma_u*1e-6:.0f} MPa  (ultimate)"),
        plt.Line2D([0], [0], color='black', marker='*', ls='None',
                   markersize=12, label='Optimum'),
    ]
    ax.set_xlabel('Bow half-angle α [°]')
    ax.set_ylabel('Bow radius R [mm]')
    ax.set_title(f'σ_max [MPa] with rigid tie  (l_S = {l_S_opt*1e3:.0f} mm)\n'
                 f'Load from 1-DOF impact model  '
                 f'(m={p.M_UAV_KG:.0f} kg, v_sink={p.V_SINK_MS:.1f} m/s)')
    ax.legend(handles=legend_lines, fontsize=8, loc='lower right')

    plt.tight_layout()
    fname = 'fig3_design_space.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ── Figure 4: Mass vs stiffness Pareto for feasible points ────────────────────

def plot_fig4(sweep: dict):
    feas = sweep['feasible']
    if not feas:
        print("  (No feasible points — skipping fig4)")
        return

    m_g   = np.array([pt['m_leg'] * 1e3 for pt in feas])
    k_kNm = np.array([pt['k_v'] * 1e-3 for pt in feas])
    sig   = np.array([pt['sigma_max'] * 1e-6 for pt in feas])

    best = sweep['best']

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(k_kNm, m_g, c=sig, cmap='plasma', s=20, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='σ_max [MPa]')
    ax.scatter(best['k_v'] * 1e-3, best['m_leg'] * 1e3,
               color='k', s=150, marker='*', zorder=5, label='Optimum')

    ax.set_xlabel('Vertical stiffness k_v [kN/m]')
    ax.set_ylabel('Half-leg mass [g]')
    ax.set_title(f'Feasible design space  ({len(feas)} points)\n'
                 'Color: peak bending stress [MPa]')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fname = 'fig4_mass_vs_stiffness.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cs = _cs()

    print_inputs(cs)

    r0_ref, r1_ref = print_reference_comparison(cs)
    print_impact(r1_ref['k_v'])

    print("  PARAMETRIC OPTIMISATION  (minimise leg mass)")
    print("-" * 72)
    sweep = run_sweep(n_R=25, n_alpha=25, n_lS=15)
    print_optimisation(sweep)

    best = sweep['best']
    print_tie_sizing(best['H_tie'])

    print("  GENERATING FIGURES")
    print("-" * 72)
    plot_fig1(best, cs)
    plot_fig2(best)
    plot_fig3(sweep)
    plot_fig4(sweep)
    print()
    print("  Done.")


if __name__ == '__main__':
    main()
