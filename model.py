"""
model.py — Castigliano structural model for a curved-skid tied-arch landing gear.

Physical setup
--------------
The landing gear consists of two symmetric legs.  Each leg is modelled as a
planar half-structure running from the foot (ground contact, s = 0) to the
crossbeam midpoint (fuselage attachment, s = s_max):

    foot ──[straight skid]──> bow-start ──[circular arc]──> apex
         ──[half crossbeam]──> symmetry plane (fuselage attach point)

Boundary conditions
-------------------
* The fuselage attachment point is assumed rigid (fixed: zero displacement,
  zero rotation).  This is justified when the airframe is much stiffer than
  the landing gear.
* The foot rests on frictionless ground: only a vertical ground-reaction F
  acts at the foot.
* A horizontal tie (cross-brace) between the two feet prevents them from
  spreading.  The tie force H is the single statically-redundant quantity.

Castigliano solution (bending-dominated, axial energy neglected)
-----------------------------------------------------------------
Bending moment at arc-length position s (free-body from foot side):

    M(s) = F · x(s) − H · y(s)

where x(s) = horizontal distance from foot, y(s) = height above ground.

Compatibility (tie prevents horizontal foot displacement):

    ∂U/∂H = 0  →  H = F · A_xy / A_yy
    with  A_xy = ∫ x y ds,  A_yy = ∫ y² ds

Vertical foot displacement (gear stroke):

    δ = F · (A_xx · A_yy − A_xy²) / (EI · A_yy)
    with  A_xx = ∫ x² ds

Assumption: axial deformation is neglected (valid when the tube is slender
relative to the arch span, i.e. I/(A·R²) ≪ 1).
"""

import numpy as np

try:
    _trapz = np.trapezoid   # NumPy ≥ 2.0
except AttributeError:
    _trapz = np.trapz


# ── Cross-section ──────────────────────────────────────────────────────────────

def cross_section(D_o: float, t_w: float) -> dict:
    """
    Compute second-moment-of-area properties for a hollow circular tube.

    Parameters
    ----------
    D_o : outer diameter [m]
    t_w : wall thickness [m]

    Returns
    -------
    dict with keys: D_o, D_i, A, I, W_b, c
    """
    D_i = D_o - 2 * t_w
    A   = np.pi / 4 * (D_o**2 - D_i**2)
    I   = np.pi / 64 * (D_o**4 - D_i**4)
    c   = D_o / 2          # distance from neutral axis to outer fibre
    W_b = I / c            # section modulus
    return dict(D_o=D_o, D_i=D_i, A=A, I=I, W_b=W_b, c=c)


# ── Geometry ───────────────────────────────────────────────────────────────────

def compute_path(R: float, alpha: float, l_S: float, L_xb: float,
                 n: int = 1500) -> list:
    """
    Discretise the half-structure into three segments.

    The path starts at the foot (s = 0) and ends at the crossbeam midpoint
    (fuselage attach, s = l_S + R·α + L_xb/2).

    Segment layout
    --------------
    skid : straight section, inclined at angle α from horizontal
           runs from foot upward to the start of the circular bow
    arc  : circular arc of radius R, from bow-start down to zero slope (apex)
    xb   : half of the horizontal crossbeam, from apex to midpoint

    Coordinate convention
    ---------------------
    x : horizontal distance from foot toward fuselage centreline  (≥ 0)
    y : height above ground                                        (≥ 0)

    Returns
    -------
    list of three segment dicts, each with keys:
        name, x, y, s, ds
    """
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)

    # ── Straight skid: foot (s=0) to bow-start ──
    s_sk = np.linspace(0.0, l_S, n)
    x_sk = s_sk * cos_a
    y_sk = s_sk * sin_a

    # ── Circular arc: bow-start to apex (theta decreases from alpha to 0) ──
    s_arc = np.linspace(0.0, R * alpha, n)
    th    = alpha - s_arc / R            # local angle, alpha → 0
    x_arc = R * (np.sin(alpha) - np.sin(th)) + l_S * cos_a
    y_arc = R * (np.cos(th) - np.cos(alpha)) + l_S * sin_a

    # ── Half crossbeam: apex to fuselage attach midpoint ──
    s_xb = np.linspace(0.0, L_xb / 2, n)
    x_xb = s_xb + R * sin_a + l_S * cos_a
    y_xb = np.full_like(s_xb, R * (1 - np.cos(alpha)) + l_S * sin_a)

    return [
        dict(name='skid', x=x_sk,  y=y_sk,  s=s_sk,            ds=l_S / (n - 1)),
        dict(name='arc',  x=x_arc, y=y_arc, s=l_S + s_arc,     ds=R * alpha / (n - 1)),
        dict(name='xb',   x=x_xb,  y=y_xb,  s=l_S + R*alpha + s_xb,
             ds=(L_xb / 2) / (n - 1)),
    ]


def gear_height(R: float, alpha: float, l_S: float) -> float:
    """Total gear height from ground to apex [m]."""
    return R * (1 - np.cos(alpha)) + l_S * np.sin(alpha)


def gear_width(R: float, alpha: float, l_S: float, L_xb: float) -> float:
    """Total track width (foot-to-foot distance) [m]."""
    return L_xb + 2 * (R * np.sin(alpha) + l_S * np.cos(alpha))


def leg_path_length(R: float, alpha: float, l_S: float, L_xb: float) -> float:
    """Arc-length of one half-leg from foot to crossbeam midpoint [m]."""
    return l_S + R * alpha + L_xb / 2


# ── Castigliano integrals ──────────────────────────────────────────────────────

def _integrate(path: list, fxy) -> float:
    """Integrate f(x, y) over all segments using the trapezoidal rule."""
    return sum(_trapz(fxy(seg['x'], seg['y']), dx=seg['ds']) for seg in path)


# ── Structural response ────────────────────────────────────────────────────────

def response(R: float, alpha: float, l_S: float, L_xb: float,
             F: float, mat, cs: dict,
             tie_mode: str = 'rigid',
             E_tie: float = 200e9, A_tie: float = 20e-6) -> dict:
    """
    Compute structural response of one half-leg under vertical load F at foot.

    Parameters
    ----------
    R, alpha, l_S : geometry design variables (radius, half-angle, skid length)
    L_xb          : crossbeam span [m]
    F             : vertical ground-reaction force at foot [N]
    mat           : Material dataclass instance
    cs            : cross-section dict (from cross_section())
    tie_mode      : 'none' | 'rigid' | 'flexible'
    E_tie, A_tie  : tie elastic modulus [Pa] and area [m²] (flexible mode only)

    Returns
    -------
    dict with keys:
        H          : tie force [N]
        delta      : vertical foot displacement (gear stroke) [m]
        k_v        : vertical stiffness [N/m]
        sigma_max  : peak bending stress along path [Pa]
        loc_max    : segment name where sigma_max occurs
        h_bow      : gear height [m]
        W_total    : track width [m]
        m_leg      : half-leg mass (foot to midpoint) [kg]
        path       : list of segment dicts (with 'M' and 'sigma' added)
        A_xx, A_xy, A_yy : Castigliano influence coefficients [m³]
    """
    EI   = mat.E * cs['I']
    path = compute_path(R, alpha, l_S, L_xb)

    A_xx = _integrate(path, lambda x, y: x * x)
    A_xy = _integrate(path, lambda x, y: x * y)
    A_yy = _integrate(path, lambda x, y: y * y)

    if tie_mode == 'none':
        H    = 0.0
        flex = A_xx / EI

    elif tie_mode == 'rigid':
        # Compatibility: ∂U/∂H = 0  →  H = F · A_xy / A_yy
        H    = F * A_xy / A_yy
        flex = (A_xx * A_yy - A_xy**2) / (EI * A_yy)

    elif tie_mode == 'flexible':
        # Tie compliance adds to the effective A_yy.
        # The tie stretches by ΔL = H · L_tie / (E_tie · A_tie).
        # Compatibility: horizontal foot displacement (from bending) + tie elongation = 0
        #   → H = F · A_xy / (A_yy + EI · L_tie / (E_tie · A_tie))
        L_tie    = gear_width(R, alpha, l_S, L_xb)
        A_yy_eff = A_yy + (EI / (E_tie * A_tie)) * L_tie
        H        = F * A_xy / A_yy_eff
        flex     = (A_xx - (H / F) * A_xy) / EI

    else:
        raise ValueError(f"Unknown tie_mode: {tie_mode!r}. "
                         "Use 'none', 'rigid', or 'flexible'.")

    delta = F * flex
    k_v   = 1.0 / flex if flex > 0 else np.inf

    # ── Bending moment and stress along path ──
    sigma_max = 0.0
    loc_max   = ''
    for seg in path:
        M     = F * seg['x'] - H * seg['y']
        sigma = M * cs['c'] / cs['I']
        seg['M']     = M
        seg['sigma'] = sigma
        sm = np.max(np.abs(sigma))
        if sm > sigma_max:
            sigma_max = sm
            loc_max   = seg['name']

    h_bow   = gear_height(R, alpha, l_S)
    W_total = gear_width(R, alpha, l_S, L_xb)
    m_leg   = mat.rho * cs['A'] * leg_path_length(R, alpha, l_S, L_xb)

    return dict(
        H=H, delta=delta, k_v=k_v,
        sigma_max=sigma_max, loc_max=loc_max,
        h_bow=h_bow, W_total=W_total, m_leg=m_leg,
        path=path,
        A_xx=A_xx, A_xy=A_xy, A_yy=A_yy,
    )


# ── Feasibility check ──────────────────────────────────────────────────────────

def is_feasible(r: dict, sigma_allow: float,
                h_range: tuple, W_range: tuple,
                delta_max: float) -> bool:
    """Return True if the design point satisfies all geometric and stress constraints."""
    return (
        h_range[0] <= r['h_bow']    <= h_range[1] and
        W_range[0] <= r['W_total']  <= W_range[1] and
        r['sigma_max'] <= sigma_allow              and
        r['delta']     <= delta_max
    )
