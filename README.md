# UAV Landing Gear — Curved Skid-Bow with Tied Arch

Structural design and optimisation of a curved-skid landing gear for a fixed-geometry UAV.  
Uses Castigliano's theorem to solve the statically-indeterminate tied-arch structure and a 1-DOF spring-mass model to derive realistic impact loads from aircraft specifications.

**HORYZN Aerostructures** / Felix Klein

---

## Problem statement

A UAV must touch down at a non-zero sink rate without exceeding material strength limits or bottoming out the gear.  The landing gear must:

- absorb the kinetic energy of touchdown elastically,
- keep peak stress within the allowable for the chosen aluminium alloy,
- fit within prescribed envelope constraints (height, track width),
- be as light as possible.

The design challenge is that gear stiffness (from geometry) and impact load (from dynamics) are coupled: a stiffer gear absorbs less stroke but produces a higher peak force, and vice versa.  This code closes that loop analytically.

---

## Physical model

### Gear geometry

Each leg consists of three segments connected in series:

```
  foot ──[straight skid]──> bow-start ──[circular arc]──> apex
       ──[half crossbeam]──> fuselage attachment (symmetry plane)
```

Design variables: bow radius `R`, bow half-angle `α`, skid length `l_S`.  
Fixed: crossbeam length `L_xb` (set by fuselage attachment spacing).

A horizontal tie (cross-brace) between the two feet prevents them from spreading outward under load.

### Structural analysis — Castigliano's theorem

The fuselage attachment is modelled as a rigid fixed support (dominant stiffness assumption).  The tie force `H` is the single statically-redundant quantity.

Bending moment at arc-length position `s` (free-body from foot side):

```
M(s) = F · x(s) − H · y(s)
```

where `x(s)` is the horizontal distance from the foot and `y(s)` is the height above ground.

Compatibility condition (tie prevents horizontal foot spread):

```
∂U/∂H = 0   →   H = F · A_xy / A_yy

A_xy = ∫ x y ds,   A_yy = ∫ y² ds   (integrated over full half-leg)
```

Vertical gear stroke (foot displacement relative to fixed apex):

```
δ = F · (A_xx · A_yy − A_xy²) / (EI · A_yy)

k_v = 1/δ̃   where δ̃ = δ/F   (stiffness from unit-load response)
```

**Assumption**: axial deformation is neglected (valid for thin-walled tubes with `I/(A·R²) ≪ 1`).

### Landing dynamics — 1-DOF spring-mass

Each leg is modelled as a linear spring with stiffness `k_v`.  At first contact, the effective mass per leg is `m = m_uav / n_legs` with initial velocity `v_sink`.

Energy balance gives the peak gear stroke:

```
δ_max = δ_st · [ 1 + √(1 + v_sink² / (g · δ_st)) ]

δ_st = m·g / k_v   (static deflection)
```

Peak ground-reaction force per leg:

```
F_impact = k_v · δ_max
```

Dynamic load factor:

```
n_dyn = F_impact / (m·g)
```

This closes the design loop: geometry → `k_v` → `F_impact` → `σ_max` → feasibility.

### Optimisation

A uniform grid over `(R, α, l_S)` is evaluated.  For each point the full chain above is executed.  The objective is **minimum half-leg mass** subject to:

| Constraint | Bound |
|---|---|
| Bending stress | σ_max ≤ σ_u / n_safety |
| Gear stroke | δ_max ≤ 80 mm |
| Gear height | 500 mm ≤ h ≤ 700 mm |
| Track width | 700 mm ≤ W ≤ 1000 mm |

---

## Repository structure

```
.
├── params.py              # All design inputs (UAV specs, material, geometry)
├── model.py               # Castigliano structural model
├── impact.py              # 1-DOF landing dynamics
├── optimize.py            # Parametric sweep and min-mass optimisation
├── run.py                 # Main entry point — runs analysis and saves plots
├── landing_gear_analysis.py  # Legacy monolithic script (v4, kept for reference)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
pip install -r requirements.txt
python run.py
```

Output figures are saved to the working directory:

| File | Content |
|---|---|
| `fig1_geometry_and_stress.png` | Gear shape, bending moment diagram, stress envelope |
| `fig2_impact_dynamics.png` | Impact force-time history and n_dyn sensitivity to sink rate |
| `fig3_design_space.png` | σ_max contour map over (R, α) with feasible region |
| `fig4_mass_vs_stiffness.png` | Pareto view of all feasible designs |

---

## Key inputs to adjust

All design parameters live in [`params.py`](params.py).  The most important ones:

| Parameter | Default | Meaning |
|---|---|---|
| `M_UAV_KG` | 80 kg | Total UAV landing mass |
| `V_SINK_MS` | 0.5 m/s | Design sink rate |
| `N_SAFETY` | 3.0 | Safety factor on σ_u |
| `D_OUTER`, `T_WALL` | 70 mm / 2.5 mm | Tube cross-section |
| `DELTA_MAX_M` | 80 mm | Stroke limit |

---

## Assumptions and limitations

- **Bending-only Castigliano**: axial deformation is neglected.  For very short, thick arches (R/D < 5) this may underestimate H and overestimate stiffness by up to ~10–15%.
- **Linear gear**: the model assumes constant stiffness throughout the stroke.  Plastic deformation and geometric nonlinearity are not captured.
- **No damping**: the 1-DOF model has zero damping, giving a conservative (upper-bound) peak force.  Real gears with rubber bushings or friction typically exhibit 10–20% lower peaks.
- **Symmetric landing**: both legs are assumed to contact simultaneously and carry equal load.  Asymmetric (one-leg) landing or lateral loads are not considered.
- **Rigid fuselage**: the airframe attachment is assumed infinitely stiff.

---

## Key findings

### The tied arch trades stress for stiffness

Without the tie, the arch is a pure bending beam with k_v ≈ 300 kN/m.  With a rigid tie, k_v rises to ≈ 3 000 kN/m — a 10× increase.  The tie reduces peak bending stress by ~60–70% but makes the structure much stiffer.

For landing gear, high stiffness means high impact loads.  The dynamic load factor for this structure is n_dyn ≈ 15–30×, which means even a slow 0.5 m/s touchdown produces F_impact ≈ 6–12 kN on an 80 kg aircraft.  At 1.0 m/s the factor exceeds 25, making the tube structurally infeasible without a cross-section redesign.

**Conclusion**: this skid-bow geometry (tied or untied) is suitable for precision VTOL landing (active height control, v_sink ≤ 0.5–0.8 m/s).  For passive landing at EASA SC-VTOL normal rates (2.5 m/s), energy-absorbing elements (rubber pads, foam crushables, oleo) would be required.

### Optimum design (80 kg, v_sink = 0.5 m/s)

| Parameter | Value |
|---|---|
| Bow radius R | 417 mm |
| Half-angle α | 69° |
| Skid length l_S | 250 mm |
| Gear height | 500 mm |
| Track width | 1207 mm |
| Vertical stiffness k_v | 2 980 kN/m |
| Dynamic load factor n_dyn | 14.9× |
| Design load F_impact | 5 865 N/leg |
| Peak stress σ_max | 153 MPa  (margin +25 % vs σ_u/n) |
| Half-leg mass | 1 306 g |

## Physical background

The **tied arch** mechanism is key to this design.  Without the tie, the arch behaves as a pure bending beam: the vertical load at the foot generates large moments at the apex, requiring a heavy cross-section.  With the tie, the horizontal foot reaction is carried in tension by the cross-brace, partially converting bending into axial load.  This reduces peak bending stress by ~60–70% for the geometries considered.

The technique is identical in principle to a tied-arch bridge: the tie carries the horizontal thrust that would otherwise require a heavy abutment.

However, the same mechanism that reduces bending stress also stiffens the structure significantly.  For a landing gear application this is a fundamental tradeoff: a stiffer gear absorbs less stroke per unit load, amplifying the dynamic impact force.  This coupling is only visible when the structural and dynamics analyses are performed together — which is the primary contribution of this code over a purely static analysis.

---

## Tech stack

- Python 3.10+
- NumPy (numerical integration, array operations)
- Matplotlib (visualisation)
