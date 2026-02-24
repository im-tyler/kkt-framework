"""
kk_dimensional_reduction.py
============================
5D -> 4D Kaluza-Klein dimensional reduction on a Klein bottle.
Careful audit of every factor of 2pi.

Central question: a0 = cH0/(2pi). Where does the 2pi come from?
Hole B: m1*c^3/(2pi*hbar) = cH0 = 2pi * a0 -- an unexplained extra 2pi.
"""

import math
from scipy.special import gamma

c    = 2.998e8
hbar = 1.0546e-34
G_N  = 6.674e-11
H0   = 2.184e-18
k_B  = 1.381e-23
a0   = 1.0421e-10
R    = c / (2 * math.pi * H0)

SEP = "=" * 72

print(SEP)
print("KK DIMENSIONAL REDUCTION: 5D -> 4D ON KLEIN BOTTLE")
print("Careful 2pi audit -- every factor tracked")
print(SEP)
print()
print(f"  c    = {c:.4e} m/s")
print(f"  hbar = {hbar:.4e} J*s")
print(f"  G_N  = {G_N:.4e} m^3/(kg*s^2)")
print(f"  H0   = {H0:.4e} s^-1")
print(f"  k_B  = {k_B:.4e} J/K")
print(f"  a0   = {a0:.4e} m/s^2")
print(f"  R    = c/(2pi*H0) = {R:.4e} m")
print(f"  Check: cH0/(2pi) = {c*H0/(2*math.pi):.4e}  a0 = {a0:.4e}")

# --- SECTION 1: Solid angles ---
print()
print(SEP)
print("SECTION 1: SOLID ANGLES IN d SPATIAL DIMENSIONS")
print(SEP)
print()
print("  Omega_d = 2*pi^(d/2) / Gamma(d/2)")

def solid_angle(d):
    return 2 * math.pi**(d/2) / gamma(d/2)

symbolic = {2:"2*pi", 3:"4*pi", 4:"2*pi^2", 5:"8*pi^2/3"}
print(f"\n  {'d':>4}  {'D=d+1':>6}  {'Omega_d':>14}  Symbolic")
print(f"  {'-'*4}  {'-'*6}  {'-'*14}  {'-'*12}")
for d in [2,3,4,5]:
    print(f"  {d:>4d}  {d+1:>6d}  {solid_angle(d):>14.6f}  {symbolic[d]}")

Omega_3 = solid_angle(3)
Omega_4 = solid_angle(4)
print(f"\n  Omega_3 = 4*pi   = {Omega_3:.6f}")
print(f"  Omega_4 = 2*pi^2 = {Omega_4:.6f}")
print(f"  Ratio Omega_4/Omega_3 = pi/2 = {Omega_4/Omega_3:.6f}  (NOT 2*pi!)")
print("  CONCLUSION: Solid angle ratio gives pi/2, NOT the missing 2pi.")

# --- SECTION 2: Green's functions ---
print()
print(SEP)
print("SECTION 2: GREEN'S FUNCTIONS AND POISSON CONVENTIONS")
print(SEP)
print("\n  G_d(r) = 1 / [(d-2)*Omega_d * r^(d-2)]")
d=3; c3 = 1.0/((d-2)*solid_angle(d))
d=4; c4 = 1.0/((d-2)*solid_angle(d))
print(f"\n  G_3(r) = 1/(4*pi*r)      coeff={c3:.6f}  expected {1/(4*math.pi):.6f}")
print(f"  G_4(r) = 1/(4*pi^2*r^2)  coeff={c4:.6f}  expected {1/(4*math.pi**2):.6f}")
print(f"  Ratio G_4/G_3 = 1/(pi*r)  [factor pi, not 2*pi]")
print(f"\n  Poisson conventions:")
print(f"    4D: nabla^2 Phi = 4*pi*G_N*rho      source factor = {4*math.pi:.4f}")
print(f"    5D: nabla^2 Phi = 4*pi^2*G_5*rho    source factor = {4*math.pi**2:.4f}")
print(f"    Ratio 5D/4D = pi = {math.pi:.4f}  (Poisson differs by pi, not 2*pi)")

# --- SECTION 3: Klein bottle compactification ---
print()
print(SEP)
print("SECTION 3: KLEIN BOTTLE COMPACTIFICATION")
print(SEP)
V_circle = 2*math.pi*R
V_KB     = math.pi*R
print(f"\n  V_S1(R)  = 2*pi*R = {V_circle:.4e} m")
print(f"  V_KB     = pi*R   = {V_KB:.4e} m")
print(f"  Ratio V_S1/V_KB = 2")
print(f"\n  KK reduction (action matching):")
print(f"    G_N^circle = G_5/(2*pi*R)")
print(f"    G_N^KB     = G_5/(pi*R)  [factor 2 larger -- less dilution]")
print(f"\n  With R = c/(2*pi*H0):  pi*R = c/(2*H0)")
print(f"    G_N^KB = 2*G_5*H0/c")
print(f"\n  In terms of a0 (H0 = 2*pi*a0/c):")
print(f"    G_N^KB = 4*pi*G_5*a0/c^2   [a0 appears naturally!]")
print(f"    G_5 = G_N * c^2/(4*pi*a0)")
print(f"\n  Ratio G_N^KB / G_N^circle = 2")
print("  Klein bottle gives TWICE Newton constant of same-R circle.")

# --- SECTION 4: KK mode spectrum ---
print()
print(SEP)
print("SECTION 4: KK MODE SPECTRUM AND COUPLINGS")
print(SEP)
print("\n  Period = 2*pi*R (FULL period before Z_2 folding)")
print("  Even modes: cos(n*y/R),  n=0,1,2,...   k_n = n/R")
print("  Odd  modes: sin(n*y/R),  n=1,2,...     k_n = n/R")
print("  Mode masses: m_n = hbar*k_n/c = n*hbar/(Rc)")
m1 = hbar/(R*c)
m1_formula = 2*math.pi*hbar*H0/c**2
print(f"\n  With R = c/(2*pi*H0):")
print(f"    m_n = n * 2*pi*hbar*H0/c^2")
print(f"    m_1 = {m1:.4e} kg  (from R)")
print(f"    m_1 = {m1_formula:.4e} kg  (= 2*pi*hbar*H0/c^2)  match={abs(m1-m1_formula)/m1<1e-4}")
print(f"    m_1*c^2 = {m1*c**2/1.602e-19:.4e} eV")
m1c3_2pi = m1*c**3/(2*math.pi*hbar)
print(f"\n  HOLE B:")
print(f"    m_1*c^3/hbar        = {m1*c**3/hbar:.4e} m/s^2  = 2*pi*cH0")
print(f"    m_1*c^3/(2*pi*hbar) = {m1c3_2pi:.4e} m/s^2  = cH0  [NOT a0!]")
print(f"    a0                  = {a0:.4e} m/s^2")
print(f"    Ratio = {m1c3_2pi/a0:.4f}  (= 2*pi = {2*math.pi:.4f})")
print(f"\n  HOLE B CONFIRMED: mode-mass route gives cH0, not a0 = cH0/(2*pi).")
print(f"  Missing factor: 2*pi still unexplained from KK alone.")

# --- SECTION 5: Five routes ---
print()
print(SEP)
print("SECTION 5: FIVE ROUTES TO a0")
print(SEP)
T_dS        = hbar*H0/(2*math.pi*k_B)
route1      = m1*c**3/(2*math.pi*hbar)
route2      = c*H0*(Omega_3/Omega_4)
a0_Unruh    = 2*math.pi*c*k_B*T_dS/hbar
a0_Verlinde = c*H0/(6*math.pi)
a0_Verlinde_c = c*H0/(2*math.pi)
a0_TGH      = c*k_B*T_dS/hbar

print(f"\n  Route 1 -- Mode mass m_1*c^3/(2*pi*hbar):")
print(f"    = {route1:.4e}  ratio/a0 = {route1/a0:.4f}  FAILS (off by 2*pi)")

print(f"\n  Route 2 -- Solid angle ratio cH0*(Omega3/Omega4):")
print(f"    = {route2:.4e}  ratio/a0 = {route2/a0:.4f}  FAILS (gives 2/pi not 1/2pi)")

print(f"\n  Route 3 -- Yukawa crossover (flat space):")
print(f"    No universal acceleration from mass-dependent Yukawa. FAILS.")

print(f"\n  Route 4 -- de Sitter + Unruh:  T_GH = hbar*H0/(2*pi*kB) = {T_dS:.4e} K")
print(f"    a_Unruh = 2*pi*c*kB*T_GH/hbar = cH0 = {a0_Unruh:.4e}")
print(f"    The 2*pi in T_GH and in Unruh formula CANCEL exactly.  FAILS.")

print(f"\n  Route 5 -- Verlinde emergent gravity:")
print(f"    Naive (6*pi): {a0_Verlinde:.4e}  ratio {a0_Verlinde/a0:.4f}  FAILS")
print(f"    N_eff=3 (2*pi): {a0_Verlinde_c:.4e}  ratio {a0_Verlinde_c/a0:.4f}  SUCCEEDS (needs dS)")

print(f"\n  *** KEY IDENTITY ***")
print(f"  a0 = c*kB*T_GH/hbar = cH0/(2*pi) = {a0_TGH:.6e}  EXACT (ratio={a0_TGH/a0:.6f})")
print(f"  This is NOT the Unruh formula (which cancels back to cH0).")
print(f"  It is: de Sitter thermal quantum  c * (k_B T_GH) / hbar")

# --- SECTION 6: 5D action ---
print()
print(SEP)
print("SECTION 6: 5D EINSTEIN-HILBERT -> 4D")
print(SEP)
G5 = G_N*math.pi*R
M4 = math.sqrt(hbar*c/G_N)
M5 = (hbar*c**3/G5)**(1/3)
print(f"\n  G_5 = G_N*pi*R = {G5:.4e} m^4/(kg*s^2)")
print(f"  M_Pl4 = sqrt(hbar*c/G_N) = {M4:.4e} kg  = {M4*c**2/1.602e-19*1e-9:.4f}e10 GeV")
print(f"  M_Pl5 = (hbar*c^3/G5)^(1/3) = {M5:.4e} kg  = {M5*c**2/1.602e-19*1e-9:.4e} GeV")
M5c = (2*H0*c)**(1/3)*M4**(2/3)
print(f"\n  M_Pl5^3 = 2*H0*c * M_Pl4^2 = 4*pi*a0 * M_Pl4^2")
print(f"  Check: {M5c:.4e} kg  match={abs(M5c-M5)/M5 < 1e-3}")

# --- SECTION 7: Summary ---
print()
print(SEP)
print("SECTION 7: 2pi AUDIT SUMMARY TABLE")
print(SEP)
print()
print(f"  {'Route / quantity':<45} {'Value':>13}  {'/ a0':>7}  Notes")
print(f"  {'-'*45} {'-'*13}  {'-'*7}  {'-'*22}")
rows = [
    ("cH0  [bare Hubble scale]",          c*H0,            c*H0/a0,           "= 2*pi*a0"),
    ("a0 = cH0/(2*pi)  [TARGET]",         a0,              1.0,               "exact"),
    ("m_1*c^3/(2*pi*hbar)  [Route 1]",    route1,          route1/a0,         "= 2*pi*a0  FAILS"),
    ("cH0*Omega3/Omega4    [Route 2]",     route2,          route2/a0,         "= (2/pi)*a0  FAILS"),
    ("cH0/(6*pi)  [Verlinde naive]",       a0_Verlinde,     a0_Verlinde/a0,    "FAILS"),
    ("cH0/(2*pi)  [Verlinde N_eff=3]",    a0_Verlinde_c,   1.0,               "CORRECT (needs dS)"),
    ("a_Unruh from T_GH   [Route 4]",     a0_Unruh,        a0_Unruh/a0,       "= 2*pi*a0  FAILS"),
    ("c*kB*T_GH/hbar  [KEY IDENTITY]",    a0_TGH,          a0_TGH/a0,         "EXACT  SUCCEEDS"),
]
for nm,v,r,note in rows:
    print(f"  {nm:<45} {v:>13.4e}  {r:>7.4f}  {note}")

print()
print(SEP)
print("FINAL VERDICT ON HOLE B")
print(SEP)
print()
print("  QUESTION: Can a0 = cH0/(2*pi) be derived from 5D KK on Klein bottle")
print("  alone (flat background)?")
print()
print("  ANSWER: NO.")
print()
print("  5D KK on Klein bottle (flat) gives:")
print("    G_N = G_5/(pi*R)   and   m_1 = 2*pi*hbar*H0/c^2")
print("    Mode mass scale: m_1*c^3/(2*pi*hbar) = cH0 = 2*pi*a0  [off by 2*pi]")
print()
print("  The missing 2*pi comes from:")
print("    a0 = c*kB*T_GH/hbar  where T_GH = hbar*H0/(2*pi*kB)")
print("    = the Gibbons-Hawking temperature of de Sitter space")
print("    = Euclidean thermal circle in de Sitter quantum gravity")
print()
print("  Klein bottle topology contributes:")
print("    V_KB = pi*R = c/(2*H0)  [factor pi from topology]")
print("    G_N^KB = 2 * G_N^circle  [non-orientability doubles G_N]")
print("    Neither supplies the missing 1/(2*pi).")
print()
print("  CLOSURE:")
print("    5D KK on Klein bottle:   G_N = G_5/(pi*R)")
print("    de Sitter background:    T_GH = hbar*H0/(2*pi*kB)")
print("    Combined:                a0 = c*kB*T_GH/hbar = cH0/(2*pi) CLOSED")
print()
print("  The 2*pi is THERMODYNAMIC (de Sitter), not topological (Klein bottle).")
print("  De Sitter background is NECESSARY to close Hole B.")
print()
print(SEP)
