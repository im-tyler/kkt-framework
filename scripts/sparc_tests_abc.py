"""
SPARC Tests A, B, C — Shape vs Scale Analysis
Tests whether KK quadrature's advantage over MOND comes from:
  A) Shape (force composition law) vs scale (a₀ value)
  B) How much of the gap closes if MOND gets a free a₀
  C) What exponent β does SPARC actually prefer?
"""

import math
import numpy as np
from scipy.optimize import minimize_scalar, minimize_scalar
import os

# ── Constants ────────────────────────────────────────────────────────────────
a0_KK   = 1.0421e-10   # m/s²  KK prediction: cH₀/(2π)
a0_MOND = 1.2e-10      # m/s²  empirical MOND

KPC_M   = 3.086e19     # m per kpc
KMS_MS  = 1e3          # (km/s → m/s)

# ── Data loading ─────────────────────────────────────────────────────────────
def load_sparc(path):
    galaxies = {}
    with open(path) as f:
        for line in f:
            if line[0] in ('#', ' ', '-', '\n'):
                continue
            cols = line.split('\t')
            if len(cols) < 7:
                continue
            try:
                name   = cols[0].strip()
                r_kpc  = float(cols[2])
                Vobs   = float(cols[3]) * KMS_MS
                eVobs  = float(cols[4]) * KMS_MS
                Vgas   = float(cols[5]) * KMS_MS
                Vdisk  = float(cols[6]) * KMS_MS
            except ValueError:
                continue
            R_m = r_kpc * KPC_M
            g_obs = Vobs**2 / R_m
            g_err = max(2 * abs(Vobs) * abs(eVobs) / R_m, 0.1 * abs(g_obs))
            g_gas  = math.copysign(Vgas**2,  Vgas)  / R_m
            g_disk = math.copysign(Vdisk**2, Vdisk) / R_m
            if name not in galaxies:
                galaxies[name] = []
            galaxies[name].append((R_m, g_obs, g_err, g_gas, g_disk))
    # Filter: need ≥3 points
    return {k: v for k, v in galaxies.items() if len(v) >= 3}

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'rotation_curves.tsv')
galaxies = load_sparc(DATA_PATH)
print(f"Loaded {len(galaxies)} galaxies, "
      f"{sum(len(v) for v in galaxies.values())} points\n")

# ── Model functions ──────────────────────────────────────────────────────────
def g_kk(g_N, a0):
    return math.sqrt(g_N**2 + g_N * a0) if g_N > 0 else g_N

def g_mond(g_N, a0):
    # Simple MOND: ν(y) = 0.5 + sqrt(0.25 + 1/y)
    if g_N <= 0:
        return g_N
    y = g_N / a0
    nu = 0.5 + math.sqrt(0.25 + 1.0 / y)
    return nu * g_N

def g_generalized(g_N, a0, beta):
    """g^(2β) = g_N^(2β) + g_N^β * a0^β  →  β=1: KK, β→0.5: MOND-like"""
    if g_N <= 0:
        return g_N
    return (g_N**(2*beta) + (g_N * a0)**beta) ** (1.0 / (2*beta))

def chi2_galaxy(data, model_fn, a0, ML):
    """χ² for one galaxy at given ML and a0."""
    total = 0.0
    for (R_m, g_obs, g_err, g_gas, g_disk) in data:
        g_N = g_gas + ML * g_disk
        pred = model_fn(g_N, a0)
        total += ((g_obs - pred) / g_err) ** 2
    return total

def best_ml_chi2(data, model_fn, a0):
    """Fit ML ∈ [0.1, 10] for a galaxy, return (ML_best, chi2_min, npts)."""
    res = minimize_scalar(
        lambda ML: chi2_galaxy(data, model_fn, a0, ML),
        bounds=(0.1, 10.0), method='bounded'
    )
    return res.x, res.fun, len(data)

# ══════════════════════════════════════════════════════════════════════════════
# TEST A — Same a₀ for both models: does KK still win on shape?
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST A — Both models at a₀ = a₀_KK = 1.0421e-10 m/s²")
print("=" * 60)

kk_wins = mond_wins = ties = 0
total_chi2_kk = total_chi2_mond = 0.0
total_pts = total_gals = 0
delta_list = []

for name, data in galaxies.items():
    _, chi2_kk,   npts = best_ml_chi2(data, g_kk,   a0_KK)
    _, chi2_mond, _    = best_ml_chi2(data, g_mond,  a0_KK)  # same a0!
    total_chi2_kk   += chi2_kk
    total_chi2_mond += chi2_mond
    total_pts       += npts
    total_gals      += 1
    delta = chi2_kk - chi2_mond   # negative = KK wins
    delta_list.append(delta)
    if   delta < -0.5: kk_wins   += 1
    elif delta >  0.5: mond_wins += 1
    else:              ties      += 1

dof_kk   = total_pts - total_gals   # 1 free param (ML) per galaxy
dof_mond = total_pts - total_gals

print(f"  KK   χ²/dof = {total_chi2_kk/dof_kk:.4f}  (total χ² = {total_chi2_kk:.1f})")
print(f"  MOND χ²/dof = {total_chi2_mond/dof_mond:.4f}  (total χ² = {total_chi2_mond:.1f})")
print(f"  Δ(total χ²) = {total_chi2_kk - total_chi2_mond:+.1f}  (negative = KK better)")
print(f"  Galaxy wins: KK {kk_wins} | MOND {mond_wins} | ties {ties}  "
      f"(KK wins {100*kk_wins/total_gals:.1f}%)")
print(f"  Mean Δχ²/galaxy = {np.mean(delta_list):+.2f}")
verdict_A = "KK shape preferred" if total_chi2_kk < total_chi2_mond else "MOND shape preferred"
print(f"  Verdict: {verdict_A}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST B — Free a₀ for MOND (global optimization)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST B — MOND with free global a₀ vs KK at fixed a₀_KK")
print("=" * 60)

def total_chi2_mond_a0(a0_val):
    tot = 0.0
    for data in galaxies.values():
        _, c2, _ = best_ml_chi2(data, g_mond, a0_val)
        tot += c2
    return tot

def total_chi2_kk_fixed():
    tot = 0.0
    for data in galaxies.values():
        _, c2, _ = best_ml_chi2(data, g_kk, a0_KK)
        tot += c2
    return tot

# Scan a0 for MOND
a0_grid = np.linspace(0.6e-10, 2.0e-10, 50)
chi2_grid = [total_chi2_mond_a0(a) for a in a0_grid]
best_idx = int(np.argmin(chi2_grid))
# Refine
res_b = minimize_scalar(
    total_chi2_mond_a0,
    bounds=(a0_grid[max(0,best_idx-2)], a0_grid[min(len(a0_grid)-1,best_idx+2)]),
    method='bounded'
)
a0_best_mond = res_b.x
chi2_mond_free = res_b.fun
chi2_kk_fixed  = total_chi2_kk_fixed()

H0_implied = a0_best_mond * 2 * math.pi / 2.998e8 * 3.086e22 / 1e3

print(f"  MOND best-fit a₀ = {a0_best_mond:.4e} m/s²")
print(f"  → implied H₀     = {H0_implied:.1f} km/s/Mpc")
print(f"  MOND(free a₀) χ² = {chi2_mond_free:.1f}  (dof={total_pts - total_gals - 1})")
print(f"  KK(fixed a₀)  χ² = {chi2_kk_fixed:.1f}  (dof={total_pts - total_gals})")
# Δχ² adjusted for extra DOF: MOND-free has 1 more free param
delta_b = chi2_mond_free - chi2_kk_fixed
print(f"  Δχ² (MOND_free - KK_fixed) = {delta_b:+.1f}")
if delta_b > 0:
    print("  Verdict: KK still beats MOND even with MOND's a₀ free (KK shape wins)")
else:
    print(f"  Verdict: MOND with free a₀ closes gap  (Δχ²={delta_b:.1f}, but +1 param)")
    print(f"  AIC penalty for extra param: MOND_free AIC advantage = {-delta_b - 2:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST C — Generalized exponent β
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST C — Best-fit exponent β  (β=1→KK, β→0.5→MOND-simple)")
print("=" * 60)

beta_grid = np.arange(0.3, 2.01, 0.05)
chi2_beta  = []

for beta in beta_grid:
    tot = 0.0
    for data in galaxies.values():
        res = minimize_scalar(
            lambda ML, b=beta: chi2_galaxy(
                data, lambda gN, a0, bb=b: g_generalized(gN, a0, bb), a0_KK, ML
            ),
            bounds=(0.1, 10.0), method='bounded'
        )
        tot += res.fun
    chi2_beta.append(tot)

chi2_beta = np.array(chi2_beta)
min_idx   = int(np.argmin(chi2_beta))
beta_best = beta_grid[min_idx]
chi2_min  = chi2_beta[min_idx]

# 95% CI: Δχ² < 3.84
ci_mask = chi2_beta < chi2_min + 3.84
ci_betas = beta_grid[ci_mask]
beta_lo  = ci_betas[0]  if len(ci_betas) else beta_best
beta_hi  = ci_betas[-1] if len(ci_betas) else beta_best

print(f"  Best-fit β = {beta_best:.2f}  (χ²_min = {chi2_min:.1f})")
print(f"  95% CI: β ∈ [{beta_lo:.2f}, {beta_hi:.2f}]")
print(f"  β=0.50 (MOND-simple): χ² = {np.interp(0.50, beta_grid, chi2_beta):.1f}  "
      f"(Δχ² = {np.interp(0.50, beta_grid, chi2_beta) - chi2_min:+.1f})")
print(f"  β=1.00 (KK quad):     χ² = {np.interp(1.00, beta_grid, chi2_beta):.1f}  "
      f"(Δχ² = {np.interp(1.00, beta_grid, chi2_beta) - chi2_min:+.1f})")

# ASCII plot of χ²(β)
print()
print("  χ²(β) curve:")
chi2_norm = chi2_beta - chi2_min
max_val   = chi2_norm[chi2_norm < 200].max() if any(chi2_norm < 200) else 50
bar_width  = 40
for i, (b, c) in enumerate(zip(beta_grid, chi2_norm)):
    bar = min(int(c / max_val * bar_width), bar_width)
    marker = ""
    if abs(b - 0.5) < 0.03:  marker = " ← MOND-simple"
    if abs(b - 1.0) < 0.03:  marker = " ← KK"
    if abs(b - beta_best) < 0.03: marker = " ← BEST FIT"
    print(f"  β={b:.2f} | {'█'*bar}{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Test A (same a₀):   KK χ²/dof={total_chi2_kk/dof_kk:.3f}, "
      f"MOND χ²/dof={total_chi2_mond/dof_mond:.3f}, "
      f"KK wins {kk_wins}/{total_gals} galaxies ({100*kk_wins/total_gals:.0f}%)")
print(f"  Test B (free a₀):   MOND best a₀={a0_best_mond:.3e} → H₀={H0_implied:.1f}; "
      f"Δχ²={delta_b:+.1f} vs KK")
print(f"  Test C (free β):    β_best={beta_best:.2f} "
      f"[CI: {beta_lo:.2f}–{beta_hi:.2f}]; "
      f"β=0.5 Δχ²={np.interp(0.50,beta_grid,chi2_beta)-chi2_min:+.1f}, "
      f"β=1.0 Δχ²={np.interp(1.00,beta_grid,chi2_beta)-chi2_min:+.1f}")

ci_includes_kk   = beta_lo <= 1.0 <= beta_hi
ci_includes_mond = beta_lo <= 0.5 <= beta_hi
print()
print(f"  95% CI includes β=1.0 (KK)?        {'YES' if ci_includes_kk else 'NO'}")
print(f"  95% CI includes β=0.5 (MOND-simple)? {'YES' if ci_includes_mond else 'NO'}")
if not ci_includes_mond and ci_includes_kk:
    print("  → DATA DISTINGUISHES: KK preferred over MOND-simple")
elif not ci_includes_kk and ci_includes_mond:
    print("  → DATA DISTINGUISHES: MOND-simple preferred over KK")
elif not ci_includes_kk and not ci_includes_mond:
    print("  → BOTH excluded: data prefers an intermediate β")
else:
    print("  → INDISTINGUISHABLE: CI includes both")
