"""
impact.py — 1-DOF linear spring-mass landing impact model.

Model description
-----------------
Each landing leg is modelled as a linear spring with vertical stiffness k_v
obtained from the Castigliano structural analysis (model.py).

At the moment of ground contact:
    position  x(0) = 0    (measured downward from first-contact point)
    velocity  ẋ(0) = v_sink

Equation of motion after contact (linear spring, no damping, rigid ground):

    m · ẍ = m · g − k_v · x

where m = m_uav / n_legs (equal load distribution assumed).

Exact solution:

    x(t) = δ_st · (1 − cos ωt) + (v_sink / ω) · sin ωt
    ω = √(k_v / m),  δ_st = m · g / k_v  (static compression)

Peak stroke (energy balance, exact for linear spring):

    δ_max = δ_st + √(δ_st² + v_sink² / ω²)
          = δ_st · [ 1 + √(1 + v_sink² / (g · δ_st)) ]

Peak ground-reaction force per leg:

    F_impact = k_v · δ_max

Dynamic load factor:

    n_dyn = F_impact / F_static = F_impact / (m · g)

Assumptions
-----------
* Linear elastic gear throughout the stroke (constant k_v).
* Rigid airframe — the fuselage does not deform.
* No aerodynamic lift or damping during impact.
* Symmetric landing: both legs contact simultaneously and carry equal load.

These assumptions are conservative (no damping → higher peak force).
"""

import numpy as np


def landing_impact(m_uav_kg: float, v_sink_ms: float,
                   k_v: float, n_legs: int = 2,
                   g: float = 9.81) -> dict:
    """
    Compute peak landing load and gear stroke from a 1-DOF impact model.

    Parameters
    ----------
    m_uav_kg  : total UAV landing mass [kg]
    v_sink_ms : sink rate at first ground contact [m/s]
    k_v       : vertical stiffness of one leg [N/m]  (from Castigliano model)
    n_legs    : number of legs sharing the load symmetrically
    g         : gravitational acceleration [m/s²]

    Returns
    -------
    dict with keys:
        m_per_leg    : mass assigned to each leg [kg]
        F_static     : quasi-static load per leg (m/n_legs · g) [N]
        delta_static : static deflection per leg [m]
        omega        : natural frequency of leg-mass system [rad/s]
        delta_max    : peak gear stroke during impact [m]
        F_impact     : peak ground-reaction force per leg [N]
        n_dyn        : dynamic load factor = F_impact / F_static [-]
        E_kinetic    : total kinetic energy at touchdown [J]
        E_gravity    : work done by gravity during peak stroke [J]
        E_elastic    : elastic strain energy stored per leg at peak stroke [J]
        energy_ok    : True if elastic capacity equals impact energy (self-check)
    """
    m = m_uav_kg / n_legs          # effective mass per leg [kg]
    F_static  = m * g              # static load per leg [N]
    delta_st  = F_static / k_v     # static deflection [m]
    omega     = np.sqrt(k_v / m)   # natural angular frequency [rad/s]

    # Peak stroke from energy balance:
    #   ½ k δ_max² = ½ m v² + m g δ_max
    #   δ_max = δ_st + √(δ_st² + v²/ω²)
    delta_max = delta_st + np.sqrt(delta_st**2 + v_sink_ms**2 / omega**2)
    F_impact  = k_v * delta_max

    n_dyn = F_impact / F_static

    # Energy terms (per leg, at peak compression)
    E_kinetic = 0.5 * m * v_sink_ms**2
    E_gravity = m * g * delta_max
    E_elastic = 0.5 * k_v * delta_max**2

    # Energy balance self-check: elastic energy should equal kinetic + gravity work
    # (tolerance allows for floating-point rounding)
    energy_ok = abs(E_elastic - (E_kinetic + E_gravity)) / (E_kinetic + 1e-9) < 1e-6

    return dict(
        m_per_leg    = m,
        F_static     = F_static,
        delta_static = delta_st,
        omega        = omega,
        delta_max    = delta_max,
        F_impact     = F_impact,
        n_dyn        = n_dyn,
        E_kinetic    = E_kinetic,
        E_gravity    = E_gravity,
        E_elastic    = E_elastic,
        energy_ok    = energy_ok,
    )


def impact_time_history(m_uav_kg: float, v_sink_ms: float,
                        k_v: float, n_legs: int = 2,
                        g: float = 9.81,
                        n_pts: int = 400) -> dict:
    """
    Return the time history of stroke and contact force over one impact cycle.

    The simulation runs from first contact until the first rebound
    (x returns to zero or one full oscillation period, whichever comes first).

    Returns
    -------
    dict with keys: t [s], x [m], F_contact [N]
    """
    m       = m_uav_kg / n_legs
    delta_st = m * g / k_v
    omega   = np.sqrt(k_v / m)

    # One full oscillation period is sufficient to capture peak + rebound
    T_period = 2 * np.pi / omega
    t = np.linspace(0.0, T_period, n_pts)

    x = delta_st * (1 - np.cos(omega * t)) + (v_sink_ms / omega) * np.sin(omega * t)

    # Contact only while x ≥ 0 (gear compressed); after rebound the model is invalid
    x_contact = np.where(x >= 0, x, np.nan)
    F_contact = np.where(x >= 0, k_v * x, np.nan)

    return dict(t=t, x=x_contact, F_contact=F_contact)
