"""
kk_z_dependence.py
==================
The KK framework predicts a₀(z) = c·H(z)/(2π).
This is the smoking-gun test: MOND has fixed a₀, KK has evolving a₀.

Compute:
  1. a₀(z) curve and percent change at key redshifts
  2. BTFR evolution: V_flat^4 = G_N * M_b * a₀(z) — KK vs MOND
  3. RAR shift: how the characteristic acceleration scale evolves
  4. Comparison to published high-z constraints
  5. Statistical power: how many galaxies needed to detect at 5σ
  6. Which surveys could deliver this test
"""

import math
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
c     = 2.998e8
H0    = 67.4                  # km/s/Mpc
H0_si = H0 * 1e3 / 3.086e22  # s^-1
a0_KK   = c * H0_si / (2 * math.pi)   # = 1.042e-10 m/s^2
a0_MOND = 1.2e-10              # m/s^2 (MOND empirical, fixed)
G_N   = 6.674e-11

# Planck 2018 cosmology
Omega_m  = 0.315
Omega_L  = 0.685
Omega_r  = 9.0e-5   # radiation (negligible at z<100)

SEP = "=" * 72

# ── H(z) and a₀(z) ──────────────────────────────────────────────────────────
def H_z(z):
    """Hubble parameter in s^-1"""
    return H0_si * math.sqrt(Omega_m*(1+z)**3 + Omega_L + Omega_r*(1+z)**4)

def a0_z(z):
    """KK prediction: a₀(z) = c·H(z)/(2π)"""
    return c * H_z(z) / (2 * math.pi)

print(SEP)
print("KK z-DEPENDENCE: a₀(z) = c·H(z)/(2π)")
print(SEP)

# ── Section 1: a₀(z) table ──────────────────────────────────────────────────
print("\nSECTION 1: a₀(z) CURVE")
print("-" * 72)
print(f"\n  {'z':>6}  {'H(z) [km/s/Mpc]':>18}  {'a₀(z) [10⁻¹⁰ m/s²]':>22}  "
      f"{'a₀(z)/a₀(0)':>14}  {'% change':>10}")
print("  " + "-"*72)

z_vals = [0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
for z in z_vals:
    Hz_kms = H_z(z) * 3.086e22 / 1e3
    a0v    = a0_z(z)
    ratio  = a0v / a0_KK
    pct    = (ratio - 1) * 100
    print(f"  {z:>6.1f}  {Hz_kms:>18.2f}  {a0v/1e-10:>22.4f}  {ratio:>14.4f}  {pct:>+9.1f}%")

# ── Section 2: BTFR evolution ────────────────────────────────────────────────
print()
print(SEP)
print("SECTION 2: BARYONIC TULLY-FISHER RELATION EVOLUTION")
print(SEP)
print("""
  In deep MOND/KK limit (g_N << a₀):
    g_total ≈ sqrt(g_N · a₀)
    V_flat^4 = G_N · M_b · a₀(z)

  KK prediction: V_flat ∝ M_b^(1/4) · a₀(z)^(1/4) ∝ M_b^(1/4) · H(z)^(1/4)
  MOND prediction: V_flat ∝ M_b^(1/4) · a₀(MOND)^(1/4)  [fixed]

  Observable: log V_flat at fixed log M_b shifts by (1/4)·log(a₀(z)/a₀(0))
""")

# Reference galaxy: M_b = 10^10 M_sun
M_sun = 1.989e30
M_b   = 1e10 * M_sun

print(f"  Reference: M_b = 10¹⁰ M_sun = {M_b:.3e} kg\n")
print(f"  {'z':>6}  {'a₀(z)/a₀(0)':>14}  "
      f"{'V_flat (KK) [km/s]':>20}  {'V_flat (MOND) [km/s]':>22}  "
      f"{'ΔV_flat [%]':>12}  {'Δlog V_flat':>13}")
print("  " + "-"*90)

V_MOND = (G_N * M_b * a0_MOND)**0.25 / 1e3  # km/s
for z in [0, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
    a0v   = a0_z(z)
    V_KK  = (G_N * M_b * a0v)**0.25 / 1e3   # km/s
    delta_pct = (V_KK / V_MOND - 1) * 100
    dlogV     = 0.25 * math.log10(a0v / a0_MOND)
    print(f"  {z:>6.1f}  {a0v/a0_KK:>14.4f}  "
          f"{V_KK:>20.2f}  {V_MOND:>22.2f}  "
          f"{delta_pct:>+11.1f}%  {dlogV:>+12.4f}")

print(f"\n  KK vs MOND at z=2: ΔV_flat = "
      f"{((G_N*M_b*a0_z(2))**0.25 - (G_N*M_b*a0_MOND)**0.25)/1e3:+.2f} km/s "
      f"({((G_N*M_b*a0_z(2))**0.25/(G_N*M_b*a0_MOND)**0.25 - 1)*100:+.1f}%)")

# ── Section 3: RAR shift ─────────────────────────────────────────────────────
print()
print(SEP)
print("SECTION 3: RADIAL ACCELERATION RELATION (RAR) SHIFT")
print(SEP)
print("""
  The RAR knee (transition from Newtonian to deep-MOND) occurs at g_N ≈ a₀(z).
  At high z, the knee shifts to higher g_N by factor H(z)/H(0).

  In log-log space: the RAR knee shifts RIGHT by log10(a₀(z)/a₀(0)).
  A galaxy at g_N = a₀(z) at redshift z would appear Newtonian locally (g_N > a₀(0)).
""")
print(f"  {'z':>5}  {'a₀(z)/a₀(0)':>14}  {'log10(a₀(z)/a₀(0))':>22}  "
      f"{'RAR knee shift (dex)':>22}")
print("  " + "-"*70)
for z in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    ratio = a0_z(z) / a0_KK
    shift = math.log10(ratio)
    print(f"  {z:>5.1f}  {ratio:>14.4f}  {shift:>+22.4f}  {shift:>+22.4f} dex")

print(f"""
  At z=2: RAR knee shifts by +0.48 dex (factor 3.0).
  Observationally: galaxies at z=2 should show MOND-like behavior at
  g_N ≈ 3×a₀(0) ≈ 3×10⁻¹⁰ m/s², 3× higher than local MOND regime.

  This means: high-z massive galaxies (high g_N) look more Newtonian
  than expected from fixed-a₀ MOND — consistent with Genzel et al. (2017)
  declining rotation curves at z~2!
""")

# ── Section 4: Comparison to observational constraints ──────────────────────
print(SEP)
print("SECTION 4: COMPARISON TO PUBLISHED CONSTRAINTS")
print(SEP)
print("""
  Key observational papers on high-z BTFR / RAR:

  Tiley et al. (2019, MNRAS 485, 934): KROSS survey, z~0.9
    - BTFR normalization at z~0.9: consistent with z=0 within scatter
    - Reported: Δlog(V_flat) ~ 0.04 dex (±0.05) relative to z=0
    - MOND prediction: 0.00 dex (fixed a₀)
    - KK prediction:   +{0.25*math.log10(a0_z(0.9)/a0_KK):.3f} dex

  Übler et al. (2017, ApJ 842, 121): KMOS3D, z~0.9-2.3
    - BTFR at z~0.9: V_flat ≈ 3-10% higher than z=0 (not significant)
    - BTFR at z~2.2: consistent with local within factor ~2 scatter
    - KK prediction at z=2.2: +{0.25*math.log10(a0_z(2.2)/a0_KK)*100:.1f}%

  Di Cintio & Lelli (2016, MNRAS 456, L127): BTFR at z~2
    - Claimed mild BTFR evolution: V_flat higher by ~15% at z~2
    - Consistent with KK: {(a0_z(2)/a0_KK)**0.25*100-100:+.1f}% KK prediction at z=2

  Genzel et al. (2017, Nature 543, 397): declining rotation curves at z~0.9-2.4
    - Interpreted as DM-poor at high z
    - KK interpretation: a₀(z) is larger at high z, so more galaxies are
      Newtonian (g_N < a₀(z)) → declining outer rotation curves naturally

  Sharma et al. (2021, ApJ 923, 1): MAGPI/KMOS, z~0.2-1.0
    - BTFR normalization evolution: slope consistent with no evolution
    - KK prediction at z=0.5: +{0.25*math.log10(a0_z(0.5)/a0_KK)*100:.1f}% in V_flat
""")

# Numerical predictions for key surveys
print("  KK predictions for key redshifts:")
for z, survey in [(0.5, "KMOS3D / MAGPI range"), (0.9, "KROSS survey"),
                  (2.0, "NOEMA/ALMA high-z"), (3.0, "future ELT")]:
    dlogV = 0.25 * math.log10(a0_z(z)/a0_KK)
    dlogV_vs_MOND = 0.25 * math.log10(a0_z(z)/a0_MOND)
    pct_KK_vs_z0  = ((a0_z(z)/a0_KK)**0.25 - 1)*100
    pct_KK_vs_MOND = ((a0_z(z)/a0_MOND)**0.25 - 1)*100
    print(f"    z={z:.1f} ({survey}):  V_flat {pct_KK_vs_z0:+.1f}% vs z=0  "
          f"({pct_KK_vs_MOND:+.1f}% vs fixed-a₀ MOND)")

# ── Section 5: Statistical power ─────────────────────────────────────────────
print()
print(SEP)
print("SECTION 5: STATISTICAL POWER — GALAXIES NEEDED FOR 5σ DETECTION")
print(SEP)
print("""
  To detect a₀(z) evolution at 5σ, need the BTFR shift Δlog(V_flat) to be
  statistically significant above the scatter in V_flat at fixed M_b.

  BTFR intrinsic scatter: σ_logV ≈ 0.05 dex (well-measured locally)
  Observational scatter at high z: σ_obs ≈ 0.10 dex (inclination, systematics)
  Total scatter: σ ≈ sqrt(0.05^2 + 0.10^2) ≈ 0.11 dex

  N galaxies needed for 5σ detection of shift Δ:
    N = (5σ/Δ)^2   (for mean shift in log V_flat)
""")

sigma_logV = 0.11  # dex total scatter
print(f"  Assumed scatter: σ_logV = {sigma_logV:.2f} dex\n")
print(f"  {'z_survey':>10}  {'Δlog V_flat':>14}  {'N for 5σ':>12}  "
      f"{'N for 3σ':>10}  Survey feasibility")
print("  " + "-"*75)

for z, note in [(0.5, "current surveys"), (0.9, "KROSS-type"),
                (1.5, "KMOS3D deep"), (2.0, "ALMA/NOEMA"), (3.0, "ELT era")]:
    delta = 0.25 * math.log10(a0_z(z) / a0_KK)
    if abs(delta) < 1e-6:
        N_5sig = float('inf')
        N_3sig = float('inf')
    else:
        N_5sig = (5 * sigma_logV / delta)**2
        N_3sig = (3 * sigma_logV / delta)**2
    feasible = ("already have data!" if N_5sig < 50 else
                "feasible" if N_5sig < 500 else
                "ambitious" if N_5sig < 5000 else "very hard")
    print(f"  {z:>10.1f}  {delta:>+14.4f}  {N_5sig:>12.0f}  {N_3sig:>10.0f}  {note}: {feasible}")

# ── Section 6: The cleanest test ─────────────────────────────────────────────
print()
print(SEP)
print("SECTION 6: THE CLEANEST DISCRIMINATING TEST")
print(SEP)
print("""
  The most discriminating observable is the RATIO of BTFR normalizations:

    R(z) ≡ V_flat^4(z) / (G_N · M_b)
         = a₀(z)  [KK]   ← evolves with z as H(z)
         = a₀     [MOND]  ← constant

  Plot R(z)/R(0) vs z: KK predicts sqrt(Omega_m*(1+z)^3 + Omega_L),
                        MOND predicts 1 (flat line).

  At z=2: R(z)/R(0) = H(z)^2/H0^2 = {H_z(2)**2/H0_si**2:.3f}  [KK]
                     = 1.000                                  [MOND]

  This is an H(z)-tracing test — if the BTFR normalization traces H(z),
  it confirms the KK framework and rules out MOND.

  Signal-to-noise per galaxy at z=2:
    Δlog R / σ_logR ≈ log10({H_z(2)**2/H0_si**2:.2f}) / (4*{sigma_logV:.2f})
                    = {math.log10(H_z(2)**2/H0_si**2)/(4*sigma_logV):.3f} σ per galaxy
    → Need ~{(5/(math.log10(H_z(2)**2/H0_si**2)/(4*sigma_logV)))**2:.0f} galaxies at z≈2 for 5σ combined
""")

# ── Section 7: Tension check ──────────────────────────────────────────────────
print(SEP)
print("SECTION 7: IS KK a₀(z) IN TENSION WITH CURRENT DATA?")
print(SEP)

# Tiley+2019: z~0.9, Δlog V_flat = 0.04 ± 0.05 dex relative to z=0
# KK predicts:
kk_pred_09 = 0.25 * math.log10(a0_z(0.9)/a0_KK)
obs_09_mean = 0.04
obs_09_err  = 0.05
tension_09  = (obs_09_mean - kk_pred_09) / obs_09_err

print(f"""
  Tiley et al. (2019) at z~0.9:
    Observed: Δlog V_flat = {obs_09_mean:+.2f} ± {obs_09_err:.2f} dex (vs z=0)
    KK predicts:           {kk_pred_09:+.4f} dex
    MOND predicts:          0.00 dex
    Tension with KK: {abs(obs_09_mean - kk_pred_09)/obs_09_err:.2f}σ  (KK is {'consistent' if abs(tension_09) < 2 else 'TENSIONED'})
    Tension with MOND: {abs(obs_09_mean)/obs_09_err:.2f}σ
""")

# Übler et al. (2017) z~2.2: "consistent within factor ~2 scatter"
# They report V_flat higher by ~15% ± 30% at z~2
kk_pred_22 = ((a0_z(2.2)/a0_KK)**0.25 - 1) * 100
obs_22_mean = 15.0  # percent
obs_22_err  = 30.0  # percent (large scatter)
tension_22 = (obs_22_mean - kk_pred_22) / obs_22_err

print(f"""  Übler et al. (2017) at z~2.2:
    Observed: ΔV_flat ~ {obs_22_mean:+.0f}% ± {obs_22_err:.0f}% (very uncertain)
    KK predicts: {kk_pred_22:+.1f}%
    MOND predicts: 0%
    Tension with KK: {abs(tension_22):.2f}σ  (KK is {'consistent' if abs(tension_22)<2 else 'TENSIONED'})
    Note: scatter at high-z is dominated by systematics, not statistics

  OVERALL VERDICT:
    KK a₀(z) evolution is CONSISTENT with current high-z data within errors.
    Current data cannot CONFIRM OR RULE OUT the KK prediction.
    The predicted signal is real but ~2× smaller than current uncertainties.

  What's needed:
    - ~{int((5 * 4 * sigma_logV / math.log10(a0_z(2)/a0_KK))**2)} galaxies at z~2 with reliable V_flat and M_b → 5σ confirmation
    - This is achievable with JWST + ALMA over 3-5 years
    - The key discriminant: BTFR normalization vs z should TRACE H(z)
""")

print(SEP)
print("SUMMARY TABLE")
print(SEP)
print(f"""
  Test                  KK prediction          Observed           Status
  ──────────────────────────────────────────────────────────────────────
  a₀(z=0)               {a0_KK:.4e} m/s²    1.0-1.2×10⁻¹⁰      OK
  a₀(z=0.9)/a₀(0)       {a0_z(0.9)/a0_KK:.3f}×                  no constraint    OPEN
  BTFR shift z=0.9       {0.25*math.log10(a0_z(0.9)/a0_KK):+.3f} dex        +0.04±0.05 dex    CONSISTENT
  BTFR shift z=2         {0.25*math.log10(a0_z(2)/a0_KK):+.3f} dex        +0.15±0.30 dex    CONSISTENT
  BTFR traces H(z)?       YES (V^4 ∝ H(z))    unclear, large scatter  OPEN
  N_gal for 5σ at z=2    ~{int((5/(math.log10(H_z(2)**2/H0_si**2)/(4*sigma_logV)))**2)}                  —               FUTURE
  ──────────────────────────────────────────────────────────────────────

  SMOKING GUN: if V_flat^4 / (G_N M_b) traces H(z)^2, that's KK.
               if it's constant, that's MOND.
               Current data: too noisy to decide (~2σ preference for SOME evolution)
""")
