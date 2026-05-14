"""
=================================================================
 UAV Landing Gear  --  Curved Skid Bow + Tied Arch  (v4)
=================================================================
HORYZN Aerostructures  /  Felix Klein

Erweitert v3 um einen horizontalen Zugverband (Tied Arch)
zwischen den beiden Fuessen einer Spange.

Wirkprinzip:
  - Ohne Tie: der Bogen ist ein reiner Biegetraeger; F_vertikal
    erzeugt grosses Moment am Apex bzw. an der Quertraversen-Mitte.
  - Mit Tie:  die Fuesse koennen sich nicht mehr nach aussen
    spreizen. Der Bogen wird zum gebundenen Bogen ("tied arch")
    -> Lastpfad geht teilweise *axial* durch die Struktur.
    Biegung wird durch H * y(s) reduziert.

Statisch unbestimmt zum Grad 1: H wird aus der Vertraeglichkeits-
bedingung u_horizontal(Fuss) = 0 bestimmt.

Castigliano:
    M(s)   = F * x_lever(s) - H * y(s)
    U      = (1/2EI) * integral( M(s)^2 ds ) ueber Skid+Bogen+Trav
    dU/dH  = 0  ->  H = F * A_xy / A_yy
                    A_xy = integral(x*y ds), A_yy = integral(y^2 ds)
    delta_apex = F * (A_xx*A_yy - A_xy^2) / (E*I*A_yy)

Wird die Tie-Steifigkeit endlich beruecksichtigt (Cable/Rod),
addiert sich (W_tie/E_t/A_t) zum effektiven A_yy:
    A_yy_eff = A_yy + (E*I/E_t/A_t) * W_total
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


# =================================================================
# 1) EINGANGSGROESSEN
# =================================================================
@dataclass
class Material:
    name:    str
    E:       float
    sigma_y: float
    sigma_u: float
    rho:     float

AL_6061_T6 = Material("Al 6061-T6", 69.0e9, 276e6, 310e6, 2700.0)
AL_7075_T6 = Material("Al 7075-T6", 71.7e9, 503e6, 572e6, 2810.0)
AL_2024_T3 = Material("Al 2024-T3", 73.1e9, 345e6, 483e6, 2780.0)

mat = AL_7075_T6

# Querschnitt der Bogen-Stange
D_o = 0.070
t_w = 0.0025

# Quertraverse + Geometrie-Constraints (aus Skizze)
L_xb        = 0.250
h_min, h_max = 0.500, 0.700
W_min, W_max = 0.700, 1.000
delta_allow  = 0.080

# Designvariablen-Bereiche
R_lo, R_hi          = 0.200, 0.600
alpha_lo, alpha_hi  = np.deg2rad(45.0), np.deg2rad(70.0)
lS_lo, lS_hi        = 0.100, 0.250

# Lastfall
F_peak    = 12_000.0
n_safety  = 3.0
n_bows    = 2

sigma_allow = mat.sigma_u / n_safety
sigma_y     = mat.sigma_y

# Initialer Punkt
R_init     = 0.403           # so gewaehlt, dass h = 500 mm bei a=70, lS=250
alpha_init = np.deg2rad(70.0)
lS_init    = 0.250

# Tie (default: ideal rigid)
TIE_MODE = 'rigid'           # 'none' | 'rigid' | 'flexible'
# Wenn flexible: Stahlseil oder Al-Stab
E_tie    = 200e9             # Stahlseil ~ 200 GPa
A_tie    = 20e-6             # 20 mm^2 (~5 mm Durchmesser)


# =================================================================
# 2) QUERSCHNITTSWERTE
# =================================================================
D_i  = D_o - 2*t_w
A_cs = np.pi/4 * (D_o**2 - D_i**2)
I_cs = np.pi/64 * (D_o**4 - D_i**4)
W_b  = I_cs / (D_o/2)
c_y  = D_o/2


# =================================================================
# 3) PFAD-KOORDINATEN UND INTEGRALE
# =================================================================
def compute_path(R, alpha, l_S, n=1500):
    """
    Halbmodell als Pfad von Foot (s=0) bis Symmetrieebene
    (s = l_S + R*alpha + L_xb/2).

    An jeder Stelle:
      x_lever = horizontale Distanz Foot -> Section (>= 0)
      y       = Hoehe ueber Boden (>= 0)
    """
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)

    # ---- Skid (s=0..l_S, vom Foot zum Bogen-Ende) ---------------
    s_sk   = np.linspace(0.0, l_S, n)
    x_sk   = s_sk*cos_a
    y_sk   = s_sk*sin_a

    # ---- Bogen (theta = alpha am Bogen-Ende .. 0 am Apex) -------
    s_arc  = np.linspace(0.0, R*alpha, n)        # vom Bogen-Ende
    th     = alpha - s_arc/R                     # theta von alpha runter
    x_arc  = R*(np.sin(alpha) - np.sin(th)) + l_S*cos_a
    y_arc  = R*(np.cos(th) - np.cos(alpha)) + l_S*sin_a

    # ---- Halbe Quertraverse (xi = 0 am Apex .. L_xb/2 in Symm) --
    s_xb   = np.linspace(0.0, L_xb/2, n)
    x_xb   = s_xb + R*sin_a + l_S*cos_a
    y_xb   = np.full_like(s_xb, R*(1-np.cos(alpha)) + l_S*sin_a)

    # Als Liste von Segmenten zurueckgeben
    return [
        dict(name='skid', x=x_sk,  y=y_sk,  s=s_sk,                    ds=l_S/(n-1)),
        dict(name='arc',  x=x_arc, y=y_arc, s=l_S + s_arc,              ds=R*alpha/(n-1)),
        dict(name='xb',   x=x_xb,  y=y_xb,  s=l_S + R*alpha + s_xb,    ds=(L_xb/2)/(n-1)),
    ]


def integrate(path, fxy):
    """Integration einer Funktion f(x,y) ueber alle Segmente."""
    return sum(_trapz(fxy(seg['x'], seg['y']), dx=seg['ds']) for seg in path)


# =================================================================
# 4) HAUPT-RESPONSE-FUNKTION  (mit/ohne Tie)
# =================================================================
def response(R, alpha, l_S, F, tie_mode='rigid', E_t=E_tie, A_t=A_tie):
    """
    Berechnet die Antwort der Halbstruktur fuer eine vertikale
    Last F am Foot.

    tie_mode:
      'none'      : keine Tie  (alte Analyse)
      'rigid'     : ideal starre Tie (H aus Castigliano-Vertraeglichkeit)
      'flexible'  : Tie mit endlicher Steifigkeit E_t * A_t
    """
    path = compute_path(R, alpha, l_S)

    A_xx = integrate(path, lambda x,y: x*x)
    A_xy = integrate(path, lambda x,y: x*y)
    A_yy = integrate(path, lambda x,y: y*y)

    if tie_mode == 'none':
        H = 0.0
        flex = A_xx/(mat.E*I_cs)
    elif tie_mode == 'rigid':
        H = F * A_xy / A_yy
        flex = (A_xx*A_yy - A_xy*A_xy)/(mat.E*I_cs*A_yy)
    elif tie_mode == 'flexible':
        # zusaetzliche Compliance der Tie:
        # u_tie = H * W_total / (E_t * A_t)
        # Vertraeglichkeit: u_horiz_foot = ... + u_tie = 0
        # -> H = F * A_xy / (A_yy + (E*I/(E_t*A_t)) * W_total)
        h_bow_v   = R*(1-np.cos(alpha)) + l_S*np.sin(alpha)
        W_total_v = L_xb + 2*(R*np.sin(alpha) + l_S*np.cos(alpha))
        A_yy_eff  = A_yy + (mat.E*I_cs/(E_t*A_t)) * W_total_v
        H = F * A_xy / A_yy_eff
        flex = (A_xx - H/F * A_xy)/(mat.E*I_cs)
    else:
        raise ValueError(tie_mode)

    delta = F * flex
    k_v   = 1.0/flex if flex > 0 else np.inf

    # Spannungsverlauf
    sigma_max = 0.0
    loc_max   = ""
    M_all, sigma_all, s_all, name_all = [], [], [], []
    for seg in path:
        M     = F*seg['x'] - H*seg['y']
        sigma = M * c_y / I_cs
        seg['M']     = M
        seg['sigma'] = sigma
        sm = np.max(np.abs(sigma))
        if sm > sigma_max:
            sigma_max = sm
            loc_max   = seg['name']
            i_max     = int(np.argmax(np.abs(sigma)))
        M_all.append(M); sigma_all.append(sigma); s_all.append(seg['s']); name_all.append(seg['name'])

    # Geometrie
    h_bow   = R*(1-np.cos(alpha)) + l_S*np.sin(alpha)
    W_total = L_xb + 2*(R*np.sin(alpha) + l_S*np.cos(alpha))
    m_leg   = mat.rho*A_cs*(L_xb/2 + R*alpha + l_S)

    return dict(
        path=path, H=H, F=F,
        sigma_max=sigma_max, location_max=loc_max,
        delta=delta, k_v=k_v,
        h_bow=h_bow, W_total=W_total, m_leg=m_leg,
        A_xx=A_xx, A_xy=A_xy, A_yy=A_yy,
    )


def feasible(r):
    return ((h_min <= r['h_bow'] <= h_max) and
            (W_min <= r['W_total'] <= W_max) and
            (r['sigma_max'] <= sigma_allow) and
            (r['delta'] <= delta_allow))


# =================================================================
# 5) AUSGABE: VERGLEICH MIT / OHNE TIE
# =================================================================
print("="*78)
print(" UAV LANDING GEAR  -  CURVED SKID-BOW + TIED ARCH  (v4)")
print("="*78)
print(f" Material            : {mat.name}    sigma_y={mat.sigma_y*1e-6:.0f}  sigma_u={mat.sigma_u*1e-6:.0f} MPa")
print(f" Querschnitt         : D_o={D_o*1e3:.1f}  t={t_w*1e3:.2f} mm   "
      f"W_b={W_b*1e9:.0f} mm^3  I={I_cs*1e12:.0f} mm^4")
print(f" L_xb                : {L_xb*1e3:.0f} mm")
print(f" Designlast F_peak   : {F_peak/1000:.1f} kN/Bein")
print(f" Auslegungs-Spannung : sigma_u/n = {sigma_allow*1e-6:.1f} MPa  (n = {n_safety:.0f})")
print(f" Initialer Punkt     : R = {R_init*1e3:.0f} mm,  alpha = {np.rad2deg(alpha_init):.0f} deg,"
      f"  l_S = {lS_init*1e3:.0f} mm")
print("-"*78)

r0 = response(R_init, alpha_init, lS_init, F_peak, tie_mode='none')
r1 = response(R_init, alpha_init, lS_init, F_peak, tie_mode='rigid')

print(f" {'':30s}  {'OHNE Tie':>14s}    {'MIT Tie (rigid)':>16s}")
print(f" {'Bauhoehe h':30s}  {r0['h_bow']*1e3:>11.1f} mm  {r1['h_bow']*1e3:>13.1f} mm")
print(f" {'Spurweite W':30s}  {r0['W_total']*1e3:>11.1f} mm  {r1['W_total']*1e3:>13.1f} mm")
print(f" {'Tie-Kraft H':30s}  {0:>11.0f}  N  {r1['H']:>13.0f}  N")
print(f" {'Apex-Durchsenkung delta':30s}  {r0['delta']*1e3:>11.2f} mm  {r1['delta']*1e3:>13.2f} mm")
print(f" {'Beinsteifigkeit k_v':30s}  {r0['k_v']*1e-3:>11.0f} kN/m {r1['k_v']*1e-3:>12.0f} kN/m")
print(f" {'Spitzen-Biegespg sigma_max':30s}  {r0['sigma_max']*1e-6:>11.1f} MPa {r1['sigma_max']*1e-6:>12.1f} MPa")
print(f" {'   Ort':30s}  {r0['location_max']:>14s}  {r1['location_max']:>16s}")
print(f" {'Margin gegen sigma_u/n=3':30s}  {(sigma_allow/r0['sigma_max']-1)*100:>+10.1f} %  {(sigma_allow/r1['sigma_max']-1)*100:>+12.1f} %")
print(f" {'Margin gegen sigma_u (n=1)':30s}  {(mat.sigma_u/r0['sigma_max']-1)*100:>+10.1f} %  {(mat.sigma_u/r1['sigma_max']-1)*100:>+12.1f} %")
print(f" {'   -> Versagensart ohne SF':30s}  {'BRUCH' if r0['sigma_max']>mat.sigma_u else 'OK':>14s}  {'BRUCH' if r1['sigma_max']>mat.sigma_u else 'OK':>16s}")
print()


# =================================================================
# 6) PARAMETRISCHE OPTIMIERUNG MIT TIE
# =================================================================
print(" 3-D OPTIMIERUNG MIT TIE (rigid)")
print("-"*78)
nR, nA, nL = 25, 25, 15
R_grid     = np.linspace(R_lo, R_hi, nR)
alpha_grid = np.linspace(alpha_lo, alpha_hi, nA)
lS_grid    = np.linspace(lS_lo, lS_hi, nL)

best_tied = None
all_pts   = []
feas_pts  = []

for R in R_grid:
    for a in alpha_grid:
        for lS in lS_grid:
            r = response(R, a, lS, F_peak, tie_mode='rigid')
            ok = feasible(r)
            pt = dict(R=R, alpha=a, lS=lS, sigma=r['sigma_max'],
                      delta=r['delta'], H=r['H'],
                      h=r['h_bow'], W=r['W_total'], m=r['m_leg'], feas=ok)
            all_pts.append(pt)
            if ok: feas_pts.append(pt)

print(f"  Suchraum:                {nR*nA*nL} Punkte")
print(f"  Davon zulaessig (n=3):   {len(feas_pts)}")

# Mit n=1 (plastische Reserve)
feas_n1 = [p for p in all_pts
           if (h_min <= p['h'] <= h_max)
           and (W_min <= p['W'] <= W_max)
           and (p['sigma'] <= mat.sigma_u)]
print(f"  Davon zulaessig (n=1):   {len(feas_n1)}  (also Strukturueberleben ohne SF)")

# Mit n=2 (eigentliche praktische Auslegung)
feas_n2 = [p for p in all_pts
           if (h_min <= p['h'] <= h_max)
           and (W_min <= p['W'] <= W_max)
           and (p['sigma'] <= mat.sigma_u/2)]
print(f"  Davon zulaessig (n=2):   {len(feas_n2)}")

if feas_pts:
    best_tied = min(feas_pts, key=lambda p: p['sigma'])
elif feas_n2:
    best_tied = min(feas_n2, key=lambda p: p['sigma'])
    print("  -> KEIN Punkt mit n=3, nehme bestes n=2-Resultat")
elif feas_n1:
    best_tied = min(feas_n1, key=lambda p: p['sigma'])
    print("  -> KEIN Punkt mit n=2, nehme bestes n=1-Resultat (Bruchgrenze)")
else:
    best_tied = min(all_pts, key=lambda p: p['sigma'])
    print("  -> KEIN Punkt erfuellt h+W-Constraint-Spannung-Kombination")

print()
print("  BEST-PUNKT:")
print(f"    R     = {best_tied['R']*1e3:.0f} mm")
print(f"    alpha = {np.rad2deg(best_tied['alpha']):.1f} deg")
print(f"    l_S   = {best_tied['lS']*1e3:.0f} mm")
print(f"    h     = {best_tied['h']*1e3:.0f} mm")
print(f"    W     = {best_tied['W']*1e3:.0f} mm")
print(f"    sigma = {best_tied['sigma']*1e-6:.1f} MPa  "
      f"(n_y={mat.sigma_y/best_tied['sigma']:.2f}  n_u={mat.sigma_u/best_tied['sigma']:.2f})")
print(f"    H     = {best_tied['H']/1000:.2f} kN  (Tie-Zug)")
print(f"    delta = {best_tied['delta']*1e3:.2f} mm")
print(f"    m/leg = {best_tied['m']*1e3:.0f} g")
print()


# =================================================================
# 7) TIE-AUSLEGUNG
# =================================================================
print(" TIE-AUSLEGUNG")
print("-"*78)
H_design = best_tied['H']
print(f"  Tie-Kraft (Zug)            : {H_design/1000:.2f} kN")

# Stahlseil (sigma_zul = 1500/2 = 750 MPa konservativ)
sigma_zul_steel = 750e6
A_min_steel     = H_design/sigma_zul_steel
d_min_steel     = np.sqrt(4*A_min_steel/np.pi)
print(f"  Stahlseil (sigma_zul=750 MPa)   : A_min = {A_min_steel*1e6:.2f} mm^2  "
      f"-> d ~ {d_min_steel*1e3:.2f} mm")

# Aluminium-Stab (Al 7075, sigma_u/3)
sigma_zul_al = mat.sigma_u/3
A_min_al     = H_design/sigma_zul_al
d_min_al     = np.sqrt(4*A_min_al/np.pi)
print(f"  Al-7075-Stab (sigma_u/3 = {sigma_zul_al*1e-6:.0f} MPa) : A_min = {A_min_al*1e6:.2f} mm^2  "
      f"-> d ~ {d_min_al*1e3:.2f} mm")

# Kohlefaser-Strang
sigma_zul_cfk = 1000e6   # konservativ fuer pultrudierte CFK-Stabe
A_min_cfk     = H_design/sigma_zul_cfk
d_min_cfk     = np.sqrt(4*A_min_cfk/np.pi)
print(f"  CFK-Stab (sigma_zul=1000 MPa)   : A_min = {A_min_cfk*1e6:.2f} mm^2  "
      f"-> d ~ {d_min_cfk*1e3:.2f} mm")
print()


# =================================================================
# 8) SENSITIVITAET sigma vs. H (Bandbreite)
# =================================================================
H_test = np.linspace(0.0, 1.5*H_design, 80)
sigma_test_apex = []
sigma_test_skid = []
sigma_test_arc  = []
sigma_test_max  = []
for H_t in H_test:
    # Kuenstliche Berechnung mit beliebigem H bei optimaler Geometrie
    R_, a_, lS_ = best_tied['R'], best_tied['alpha'], best_tied['lS']
    path = compute_path(R_, a_, lS_)
    smax_seg = {}
    for seg in path:
        M = F_peak*seg['x'] - H_t*seg['y']
        smax_seg[seg['name']] = np.max(np.abs(M*c_y/I_cs))
    sigma_test_apex.append(smax_seg['xb'])
    sigma_test_skid.append(smax_seg['skid'])
    sigma_test_arc.append(smax_seg['arc'])
    sigma_test_max.append(max(smax_seg.values()))


# =================================================================
# 9) PLOTS
# =================================================================
# Plot A: Vergleich Spannungsverlauf mit/ohne Tie
fig, axs = plt.subplots(1, 3, figsize=(17, 5))

# (a) Geometrie der Spange
def plot_bow(ax, R, alpha, l_S, color='k'):
    path = compute_path(R, alpha, l_S, n=400)
    skid, arc, xb = path
    # Globale Position: foot bei +x = +(L_xb/2 + R sin a + l_S cos a), spiegeln
    Wt = L_xb + 2*(R*np.sin(alpha) + l_S*np.cos(alpha))
    foot_x = Wt/2
    # rechte Haelfte:
    x_sk = foot_x - skid['x'];  y_sk = skid['y']
    x_ar = foot_x - arc['x'];   y_ar = arc['y']
    x_xb = foot_x - xb['x'];    y_xb = xb['y']
    ax.plot( x_sk, y_sk, 'r-', lw=2)
    ax.plot(-x_sk, y_sk, 'r-', lw=2)
    ax.plot( x_ar, y_ar, 'g-', lw=2.5)
    ax.plot(-x_ar, y_ar, 'g-', lw=2.5)
    ax.plot([-L_xb/2, L_xb/2], [xb['y'][0], xb['y'][0]], 'b-', lw=3)
    # Tie
    ax.plot([-foot_x+0.005, foot_x-0.005], [0, 0], color='purple', lw=1.5,
            ls='--', label='Tie (Zugverband)')
    ax.plot([-foot_x, foot_x], [-0.005, -0.005], 'k-', lw=0.5)
    return foot_x, xb['y'][0]

best = best_tied
fx, hb = plot_bow(axs[0], best['R'], best['alpha'], best['lS'])
axs[0].axhline(0, color='gray', ls=':', lw=0.8)
axs[0].axvline(0, color='gray', ls=':', lw=0.8)
axs[0].set_xlabel('x [m]'); axs[0].set_ylabel('y [m]')
axs[0].set_title(f"Optimum-Spange (mit Tie)\n"
                 f"R={best['R']*1e3:.0f} mm, $\\alpha$={np.rad2deg(best['alpha']):.0f}°, "
                 f"l_S={best['lS']*1e3:.0f} mm\n"
                 f"H = {best['H']/1000:.1f} kN")
axs[0].set_aspect('equal'); axs[0].grid(alpha=0.3); axs[0].legend(loc='upper right', fontsize=8)

# (b) Biegemoment laengs Pfad: ohne vs mit Tie
def collect_along_path(path, key):
    s = np.concatenate([seg['s'] for seg in path])
    v = np.concatenate([seg[key] for seg in path])
    return s, v

r_no  = response(best['R'], best['alpha'], best['lS'], F_peak, tie_mode='none')
r_yes = response(best['R'], best['alpha'], best['lS'], F_peak, tie_mode='rigid')

s_n, M_n = collect_along_path(r_no['path'],  'M')
s_y, M_y = collect_along_path(r_yes['path'], 'M')
s_n2, sg_n = collect_along_path(r_no['path'], 'sigma')
s_y2, sg_y = collect_along_path(r_yes['path'],'sigma')

# Segmentgrenzen
seg_bound1 = best['lS']
seg_bound2 = best['lS'] + best['R']*best['alpha']

axs[1].plot(s_n*1e3,  M_n,  'r-',  lw=2, label='OHNE Tie')
axs[1].plot(s_y*1e3,  M_y,  'g-',  lw=2, label='MIT Tie (rigid)')
axs[1].axhline(0, color='gray', lw=0.5)
axs[1].axvline(seg_bound1*1e3, color='gray', ls=':', label='Skid->Bogen')
axs[1].axvline(seg_bound2*1e3, color='black', ls=':', label='Bogen->Quertraverse')
axs[1].set_xlabel('Pfadlaenge s vom Boden [mm]')
axs[1].set_ylabel('M [Nm]')
axs[1].set_title('Biegemoment laengs Halbstruktur')
axs[1].grid(alpha=0.3); axs[1].legend(fontsize=9, loc='lower left')

axs[2].plot(s_n2*1e3, sg_n*1e-6, 'r-', lw=2, label='OHNE Tie')
axs[2].plot(s_y2*1e3, sg_y*1e-6, 'g-', lw=2, label='MIT Tie (rigid)')
axs[2].axhline( sigma_allow*1e-6, color='red',    ls='--', alpha=0.6, label=f'$\\sigma_u/3$ = {sigma_allow*1e-6:.0f}')
axs[2].axhline(-sigma_allow*1e-6, color='red',    ls='--', alpha=0.6)
axs[2].axhline( mat.sigma_u/2*1e-6, color='orange', ls='--', alpha=0.5, label=f'$\\sigma_u/2$ = {mat.sigma_u/2*1e-6:.0f}')
axs[2].axhline(-mat.sigma_u/2*1e-6, color='orange', ls='--', alpha=0.5)
axs[2].axhline( mat.sigma_u*1e-6,   color='black',  ls=':',  alpha=0.6, label=f'$\\sigma_u$ = {mat.sigma_u*1e-6:.0f}')
axs[2].axhline(-mat.sigma_u*1e-6,   color='black',  ls=':',  alpha=0.6)
axs[2].axvline(seg_bound1*1e3, color='gray', ls=':')
axs[2].axvline(seg_bound2*1e3, color='black', ls=':')
axs[2].set_xlabel('Pfadlaenge s vom Boden [mm]')
axs[2].set_ylabel(r'$\sigma$ [MPa]')
axs[2].set_title('Biegespannung laengs Halbstruktur')
axs[2].grid(alpha=0.3); axs[2].legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig('landing_gear_response.png', dpi=160)


# Plot B: sigma_max vs H sweep + Heatmap mit Tie
fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5))

axs2[0].plot(H_test/1000, np.array(sigma_test_max)*1e-6, 'k-',  lw=2.5, label='max ueber Pfad')
axs2[0].plot(H_test/1000, np.array(sigma_test_apex)*1e-6, 'b--', lw=1.5, label='Quertraverse')
axs2[0].plot(H_test/1000, np.array(sigma_test_arc)*1e-6,  'g--', lw=1.5, label='Bogen')
axs2[0].plot(H_test/1000, np.array(sigma_test_skid)*1e-6, 'r--', lw=1.5, label='Skid')
axs2[0].axvline(H_design/1000, color='purple', ls=':', label=f'H_compat = {H_design/1000:.1f} kN')
axs2[0].axhline(sigma_allow*1e-6, color='red',    ls='--', alpha=0.5, label='$\\sigma_u/3$')
axs2[0].axhline(mat.sigma_u/2*1e-6, color='orange', ls='--', alpha=0.5, label='$\\sigma_u/2$')
axs2[0].axhline(mat.sigma_u*1e-6,   color='black',  ls=':',  alpha=0.5, label='$\\sigma_u$')
axs2[0].set_xlabel('Tie-Kraft H [kN]')
axs2[0].set_ylabel(r'$\sigma_{\max}$ [MPa]')
axs2[0].set_title(f'Sensitivitaet: $\\sigma_{{\\max}}$ vs Tie-Kraft H\n'
                  f'(R={best["R"]*1e3:.0f}, $\\alpha$={np.rad2deg(best["alpha"]):.0f}°, '
                  f'l_S={best["lS"]*1e3:.0f})')
axs2[0].grid(alpha=0.3); axs2[0].legend(fontsize=8, loc='upper right')

# Heatmap (R, alpha) bei l_S = best['lS'] mit Tie
A_deg, R_mm = np.meshgrid(np.rad2deg(alpha_grid), R_grid*1e3)
sigma_field = np.zeros((nR, nA))
ok_field    = np.zeros((nR, nA), dtype=bool)
for i, R in enumerate(R_grid):
    for j, a in enumerate(alpha_grid):
        rr = response(R, a, best['lS'], F_peak, tie_mode='rigid')
        sigma_field[i,j] = rr['sigma_max']*1e-6
        ok_field[i,j] = feasible(rr)

cs = axs2[1].contourf(A_deg, R_mm, sigma_field, 25, cmap='viridis')
axs2[1].contour(A_deg, R_mm, sigma_field, levels=[sigma_allow*1e-6],
                colors='red', linewidths=2, linestyles='--')
axs2[1].contour(A_deg, R_mm, sigma_field, levels=[mat.sigma_u/2*1e-6],
                colors='orange', linewidths=2, linestyles='--')
axs2[1].contour(A_deg, R_mm, sigma_field, levels=[mat.sigma_u*1e-6],
                colors='white', linewidths=1.5, linestyles=':')
axs2[1].plot(np.rad2deg(best['alpha']), best['R']*1e3, 'k*',
             markersize=20, label='Optimum')
axs2[1].set_xlabel(r'$\alpha$ [deg]'); axs2[1].set_ylabel('R [mm]')
axs2[1].set_title(f'$\\sigma_{{\\max}}$ [MPa] mit Tie  (l_S = {best["lS"]*1e3:.0f} mm)')
plt.colorbar(cs, ax=axs2[1])
axs2[1].legend()

plt.tight_layout()
plt.savefig('landing_gear_parametric.png', dpi=160)

print("Plots gespeichert:")
print("  - landing_gear_response.png")
print("  - landing_gear_parametric.png")
