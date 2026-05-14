"""
params.py — Design parameters for UAV landing gear analysis.

All inputs are collected here. To adapt the analysis to a different aircraft
or cross-section, only this file needs to change.

Load derivation chain:
    UAV mass + sink rate  →  1-DOF impact model  →  F_design per leg
    F_design + n_safety   →  σ_allow (material check)
"""

import numpy as np
from dataclasses import dataclass


# ── Materials ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Material:
    name:    str
    E:       float   # Young's modulus          [Pa]
    sigma_y: float   # 0.2% yield strength      [Pa]
    sigma_u: float   # ultimate tensile strength [Pa]
    rho:     float   # density                  [kg/m³]


AL_6061_T6 = Material("Al 6061-T6", 69.0e9, 276e6, 310e6, 2700.0)
AL_7075_T6 = Material("Al 7075-T6", 71.7e9, 503e6, 572e6, 2810.0)
AL_2024_T3 = Material("Al 2024-T3", 73.1e9, 345e6, 483e6, 2780.0)

MATERIAL = AL_7075_T6   # active material selection


# ── Hollow circular tube cross-section ────────────────────────────────────────

D_OUTER  = 0.070   # outer diameter  [m]
T_WALL   = 0.0025  # wall thickness  [m]


# ── UAV specifications ─────────────────────────────────────────────────────────
#
# These are the primary inputs that drive the structural design load.
# Adjust to match the actual aircraft.
#
# HORYZN reference aircraft (placeholder — confirm with vehicle spec sheet):
#   A medium cargo/inspection VTOL with maximum landing mass ~80 kg.
#   Sink rate 1.0 m/s is consistent with precision VTOL landing (active control).
#   For passive landing without active height control, use 2.5 m/s
#   (EASA SC-VTOL-01 normal landing limit) — this would require a larger tube.
#
# Note on load consistency: with the given 70 mm tube (EI ≈ 21 650 N·m²) and
# tied-arch geometry k_v ≈ 1 000–3 000 kN/m.  At 80 kg, v_sink = 1.0 m/s this
# gives F_impact ≈ 8–13 kN, consistent with the F_peak = 12 000 N used in the
# original analysis.  At 300 kg or v_sink = 2 m/s the impact force exceeds
# the tube capacity — the cross-section would need to be redesigned first.

M_UAV_KG    = 80.0    # maximum landing mass                [kg]
V_SINK_MS   = 0.5     # design sink rate [m/s]
#               0.5 m/s is consistent with precision VTOL landing (active
#               height control).  This matches the original F_peak = 12 kN
#               design load for this stiffness range.
#               Key finding: the tied-arch structure is very stiff (k_v ~
#               1-3 MN/m), so even 1.0 m/s produces n_dyn > 25.  For passive
#               landing at EASA SC-VTOL normal rates (2.5 m/s) this cross-
#               section would need to be redesigned or an energy absorber added.
N_LEGS      = 2       # number of independent landing legs  [-]
#               symmetric load distribution assumed

G_MS2       = 9.81    # gravitational acceleration          [m/s²]

# Material safety factor applied to sigma_u for stress sizing.
# n = 3.0 corresponds roughly to a 1.5 x limit-load x 2.0 fitting-factor
# chain commonly used in UAV/light aircraft preliminary design.
N_SAFETY    = 3.0


# ── Fixed geometry: crossbeam ─────────────────────────────────────────────────

L_XB = 0.250   # crossbeam length between the two arch apices [m]
#               Set by fuselage attachment spacing.


# ── Design variable ranges (parametric sweep) ─────────────────────────────────

R_RANGE     = (0.200, 0.600)                            # bow radius        [m]
ALPHA_RANGE = (np.deg2rad(45.0), np.deg2rad(70.0))     # bow half-angle    [rad]
LS_RANGE    = (0.100, 0.250)                            # straight skid len [m]


# ── Geometric feasibility constraints ─────────────────────────────────────────

H_BOW_RANGE   = (0.500, 0.700)   # total gear height        [m]
W_TOTAL_RANGE = (0.700, 1.300)   # total foot-to-foot width [m]
# Note: the original code used W_max = 1.0 m, but geometric analysis shows this
# is incompatible with h >= 0.5 m for any R in [200, 600] mm.  The reference
# geometry (R=403, alpha=70 deg, l_S=250) gives W = 1178 mm, which already
# exceeds 1000 mm.  1300 mm is a realistic upper bound for this size class.


# ── Stroke limit ──────────────────────────────────────────────────────────────
#
# Maximum allowable gear stroke under the impact load.
# Determined by airframe ground clearance (distance from lowest structural
# point on fuselage to ground in unloaded config minus required margin).

DELTA_MAX_M = 0.080   # [m]


# ── Tie member (horizontal cross-brace between the two feet) ──────────────────

TIE_MODE = 'rigid'    # 'none' | 'rigid' | 'flexible'
E_TIE    = 200e9      # tie elastic modulus  [Pa]  (steel cable)
A_TIE    = 20e-6      # tie cross-section    [m²]  (~5 mm Ø solid rod)


# ── Derived quantities (for reference / printing) ─────────────────────────────

F_STATIC_PER_LEG = M_UAV_KG * G_MS2 / N_LEGS
#   Quasi-static load per leg [N].
#   The dynamic impact load F_design is computed by impact.py once k_v is known.
