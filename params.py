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
#   A large multirotor/VTOL with maximum landing mass ~300 kg,
#   normal landing sink rate 2.0 m/s per EASA SC-VTOL-01 §CS-23.473.

M_UAV_KG    = 300.0   # maximum landing mass                [kg]
V_SINK_MS   = 2.0     # design sink rate (normal landing)   [m/s]
#               EASA SC-VTOL-01: normal ≤ 2.5 m/s, hard landing ≤ 3.5 m/s
N_LEGS      = 2       # number of independent landing legs  [-]
#               symmetric load distribution assumed

G_MS2       = 9.81    # gravitational acceleration          [m/s²]

# Material safety factor applied to σ_u for stress sizing.
# n = 3.0 corresponds roughly to a 1.5 × limit-load × 2.0 fitting-factor
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

H_BOW_RANGE   = (0.500, 0.700)   # total gear height   [m]
W_TOTAL_RANGE = (0.700, 1.000)   # total track width   [m]


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
