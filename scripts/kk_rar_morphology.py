"""
kk_rar_morphology.py
RAR analysis binned by galaxy surface brightness class.

Question: Does KK beat MOND more in transition-zone galaxies (y~0.5-3)?
Prediction: Yes — KK has functional shape advantage in transition, not deep MOND.

Classification proxy: median log(g_bar) per galaxy
  LSB  = low surface brightness  → low g_bar, deep MOND dominated
  MED  = medium                  → y~0.5-3, transition zone
  HSB  = high surface brightness → high g_bar, mostly Newtonian

Panels:
  A: Win rate (KK vs MOND) by galaxy class
  B: Mean Δχ² = χ²_MOND − χ²_KK per acceleration bin, split by class
  C: RAR scatter coloured by class
  D: Cumulative Δχ² vs y threshold — shows where KK advantage lives
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

# ─── constants ────────────────────────────────────────────────────────────────
H0   = 67.4e3 / 3.0856e22          # s⁻¹
c    = 3e8                          # m/s
A0   = c * H0 / (2 * np.pi)        # 1.0421e-10 m/s²
kpc  = 3.0856e19                    # m

# ─── model functions ──────────────────────────────────────────────────────────
def nu_KK(y):
    return np.sqrt(1.0 + 1.0 / np.where(y > 0, y, 1e-30))

def nu_MOND(y):
    return 0.5 + np.sqrt(0.25 + 1.0 / np.where(y > 0, y, 1e-30))

def nu_GAUGE(y):
    return 1.0 + 1.0 / np.sqrt(np.where(y > 0, y, 1e-30))

# ─── parse SPARC ──────────────────────────────────────────────────────────────
def parse_sparc(path):
    galaxies = {}
    with open(path) as f:
        for line in f:
            if not line.strip() or line[0] in ('#', ' ', '-'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                name = parts[0]
                R    = float(parts[2]) * kpc
                Vobs = float(parts[3]) * 1e3
                eVobs= float(parts[4]) * 1e3
                Vgas = float(parts[5]) * 1e3
                Vdisk= float(parts[6]) * 1e3
            except ValueError:
                continue
            if R <= 0 or eVobs <= 0 or Vobs <= 0:
                continue
            g_obs = Vobs**2 / R
            g_err = 2 * abs(Vobs) * eVobs / R
            # g_bar from gas + disk (baryonic, ML free)
            ggas  = np.sign(Vgas)  * Vgas**2  / R
            gdisk = np.sign(Vdisk) * Vdisk**2 / R
            if name not in galaxies:
                galaxies[name] = []
            galaxies[name].append((g_obs, g_err, ggas, gdisk, R))
    return galaxies

# ─── per-galaxy χ² fit ────────────────────────────────────────────────────────
def fit_galaxy(pts, nu_fn):
    """Fit ML (mass-to-light for disk) to minimise χ². Gas is fixed."""
    g_obs_arr = np.array([p[0] for p in pts])
    g_err_arr = np.array([p[1] for p in pts])
    ggas_arr  = np.array([p[2] for p in pts])
    gdisk_arr = np.array([p[3] for p in pts])

    def chi2(ML):
        if ML < 0:
            return 1e10
        g_bar = ggas_arr + ML * gdisk_arr
        # only points where g_bar > 0 (physical)
        mask  = g_bar > 0
        if mask.sum() < 2:
            return 1e10
        y     = g_bar[mask] / A0
        g_pred= nu_fn(y) * g_bar[mask]
        res   = (g_obs_arr[mask] - g_pred) / g_err_arr[mask]
        return np.sum(res**2)

    result = minimize_scalar(chi2, bounds=(0.1, 10.0), method='bounded')
    ML_best = result.x
    chi2_best = result.fun

    # compute per-point residuals at best ML for binned analysis
    g_bar = ggas_arr + ML_best * gdisk_arr
    mask  = g_bar > 0
    y_pts  = g_bar[mask] / A0
    g_pred = nu_fn(y_pts) * g_bar[mask]
    resid  = (g_obs_arr[mask] - g_pred) / g_err_arr[mask]
    npts   = mask.sum()
    dof    = max(npts - 1, 1)
    return chi2_best / dof, ML_best, y_pts, resid, g_obs_arr[mask], g_bar[mask]

# ─── main ─────────────────────────────────────────────────────────────────────
path = 'data/rotation_curves.tsv'
galaxies = parse_sparc(path)
print(f"Loaded {len(galaxies)} galaxies")

results = []
for name, pts in galaxies.items():
    if len(pts) < 5:
        continue
    chi2_KK,  ML_KK,  y_KK,  r_KK,  gobs_KK,  gbar_KK  = fit_galaxy(pts, nu_KK)
    chi2_M,   ML_M,   y_M,   r_M,   gobs_M,   gbar_M   = fit_galaxy(pts, nu_MOND)
    chi2_G,   ML_G,   y_G,   r_G,   gobs_G,   gbar_G   = fit_galaxy(pts, nu_GAUGE)

    # proxy for surface brightness: median log10(g_bar) at best KK ML
    g_bar_arr = np.array([p[2] + ML_KK*p[3] for p in pts])
    g_bar_arr = g_bar_arr[g_bar_arr > 0]
    if len(g_bar_arr) == 0:
        continue
    med_log_gbar = np.median(np.log10(g_bar_arr))

    results.append({
        'name': name,
        'chi2_KK': chi2_KK, 'chi2_MOND': chi2_M, 'chi2_GAUGE': chi2_G,
        'med_log_gbar': med_log_gbar,
        'y_pts': y_KK, 'gobs_pts': gobs_KK, 'gbar_pts': gbar_KK,
        'npts': len(y_KK)
    })

print(f"Fitted {len(results)} galaxies with ≥5 points")

# ─── classify by surface brightness ──────────────────────────────────────────
log_g = np.array([r['med_log_gbar'] for r in results])
p33, p67 = np.percentile(log_g, 33), np.percentile(log_g, 67)
print(f"log10(g_bar) percentiles: 33%={p33:.2f}, 67%={p67:.2f}")

def classify(lg):
    if lg < p33:  return 'LSB'
    if lg < p67:  return 'MED'
    return 'HSB'

for r in results:
    r['class'] = classify(r['med_log_gbar'])

classes = ['LSB', 'MED', 'HSB']
colors  = {'LSB': '#4488FF', 'MED': '#FFAA33', 'HSB': '#FF4444'}
labels  = {'LSB': 'LSB (low $g_{bar}$)', 'MED': 'Medium', 'HSB': 'HSB (high $g_{bar}$)'}

# ─── win rates ────────────────────────────────────────────────────────────────
print("\n=== Win rates by class ===")
for cls in classes:
    sub = [r for r in results if r['class'] == cls]
    n   = len(sub)
    kk_wins   = sum(1 for r in sub if r['chi2_KK']   < r['chi2_MOND'] and r['chi2_KK']   < r['chi2_GAUGE'])
    mond_wins = sum(1 for r in sub if r['chi2_MOND']  < r['chi2_KK']  and r['chi2_MOND']  < r['chi2_GAUGE'])
    gaug_wins = sum(1 for r in sub if r['chi2_GAUGE'] < r['chi2_KK']  and r['chi2_GAUGE'] < r['chi2_MOND'])
    dchi2_mean = np.mean([r['chi2_MOND'] - r['chi2_KK'] for r in sub])
    print(f"  {cls:3s} (n={n:3d}): KK={kk_wins/n*100:.1f}%  MOND={mond_wins/n*100:.1f}%  "
          f"GAUGE={gaug_wins/n*100:.1f}%  mean(Δχ²)={dchi2_mean:+.3f}")

# ─── acceleration bin analysis ────────────────────────────────────────────────
# Collect all points with their class label
all_y, all_dchi2_pt, all_cls = [], [], []

for r in results:
    # For each point, compute contribution to Δχ²/dof
    # Approximate: use per-galaxy Δχ²/dof as the bin signal
    # More directly: use (r_MOND² - r_KK²) per point
    pts = galaxies[r['name']]
    pts_clean = [(p[0], p[1], p[2], p[3]) for p in pts if p[0]>0 and p[1]>0]
    if not pts_clean:
        continue

    # Recompute per-point residuals at best ML for each model
    g_obs_a = np.array([p[0] for p in pts_clean])
    g_err_a = np.array([p[1] for p in pts_clean])
    ggas_a  = np.array([p[2] for p in pts_clean])
    gdisk_a = np.array([p[3] for p in pts_clean])

    # Use the best ML from the fits stored above
    # Re-fit quickly
    for nu_fn, key in [(nu_KK, 'KK'), (nu_MOND, 'MOND')]:
        def chi2_fn(ML, g_obs=g_obs_a, g_err=g_err_a, ggas=ggas_a, gdisk=gdisk_a):
            g_bar = ggas + ML * gdisk
            mask  = g_bar > 0
            if mask.sum() < 2:
                return 1e10
            y     = g_bar[mask] / A0
            g_pred= nu_fn(y) * g_bar[mask]
            return np.sum(((g_obs[mask]-g_pred)/g_err[mask])**2)
        res = minimize_scalar(chi2_fn, bounds=(0.1,10.0), method='bounded')
        ML  = res.x
        g_bar = ggas_a + ML * gdisk_a
        mask  = g_bar > 0
        y_pts = (g_bar/A0)[mask]
        g_pred= nu_fn(y_pts) * g_bar[mask]
        resid = ((g_obs_a[mask]-g_pred)/g_err_a[mask])**2
        if key == 'KK':
            chi2_KK_pts  = resid
            y_KK_pts     = y_pts
            g_bar_KK_pts = g_bar[mask]
            gobs_KK_pts  = g_obs_a[mask]
        else:
            chi2_MOND_pts = resid

    for i in range(len(y_KK_pts)):
        # Δχ²_i = χ²_MOND_i - χ²_KK_i (positive = KK better)
        if i < len(chi2_MOND_pts):
            all_y.append(y_KK_pts[i])
            all_dchi2_pt.append(chi2_MOND_pts[i] - chi2_KK_pts[i])
            all_cls.append(r['class'])

all_y       = np.array(all_y)
all_dchi2   = np.array(all_dchi2_pt)
all_cls_arr = np.array(all_cls)

print(f"\nTotal points for bin analysis: {len(all_y)}")

# ─── make figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#0A0A1A')
for ax in axes.flat:
    ax.set_facecolor('#111122')
    ax.tick_params(colors='#CCCCEE')
    ax.spines['bottom'].set_color('#333366')
    ax.spines['top'].set_color('#333366')
    ax.spines['left'].set_color('#333366')
    ax.spines['right'].set_color('#333366')

# ── Panel A: Win rates by class ───────────────────────────────────────────────
ax = axes[0, 0]
cls_names = classes
kk_rates   = []
mond_rates = []
for cls in cls_names:
    sub = [r for r in results if r['class'] == cls]
    n   = len(sub)
    kk_rates.append(  sum(1 for r in sub if r['chi2_KK']  < r['chi2_MOND']) / n * 100)
    mond_rates.append(sum(1 for r in sub if r['chi2_MOND'] < r['chi2_KK'])  / n * 100)

x = np.arange(3)
w = 0.32
bars_kk   = ax.bar(x - w/2, kk_rates,   w, color='#4ECDC4', alpha=0.9, label='KK wins')
bars_mond = ax.bar(x + w/2, mond_rates,  w, color='#FF6B35', alpha=0.9, label='MOND wins')
ax.axhline(50, color='#555577', ls='--', lw=1)
ax.set_xticks(x)
ax.set_xticklabels([labels[c] for c in cls_names], color='#CCCCEE', fontsize=9)
ax.set_ylabel('% of galaxies', color='#CCCCEE')
ax.set_title('Panel A: Win Rate by Surface Brightness Class', color='#CCCCEE', fontsize=10)
ax.legend(fontsize=9, facecolor='#111122', labelcolor='#CCCCEE')
ax.set_ylim(0, 85)
for i, (kk, mo) in enumerate(zip(kk_rates, mond_rates)):
    ax.text(i-w/2, kk+1.5, f'{kk:.0f}%', ha='center', fontsize=8, color='#4ECDC4')
    ax.text(i+w/2, mo+1.5, f'{mo:.0f}%', ha='center', fontsize=8, color='#FF6B35')

# ── Panel B: Mean Δχ² per y-bin by class ─────────────────────────────────────
ax = axes[0, 1]
y_edges = np.logspace(-2, 2, 18)
y_mids  = 0.5 * (y_edges[:-1] + y_edges[1:])

for cls in classes:
    mask_cls = all_cls_arr == cls
    y_c   = all_y[mask_cls]
    dc_c  = all_dchi2[mask_cls]
    mean_dc = []
    for j in range(len(y_edges)-1):
        bin_mask = (y_c >= y_edges[j]) & (y_c < y_edges[j+1])
        if bin_mask.sum() > 3:
            mean_dc.append(np.mean(dc_c[bin_mask]))
        else:
            mean_dc.append(np.nan)
    mean_dc = np.array(mean_dc)
    valid = ~np.isnan(mean_dc)
    ax.plot(y_mids[valid], mean_dc[valid], 'o-', color=colors[cls],
            label=labels[cls], lw=1.5, ms=4, alpha=0.85)

ax.axhline(0, color='#555577', ls='--', lw=1)
ax.axvspan(0.5, 3.0, alpha=0.08, color='#88FF88', label='Transition zone y=0.5–3')
ax.set_xscale('log')
ax.set_xlabel('y = g_bar / a₀', color='#CCCCEE')
ax.set_ylabel('Mean(χ²_MOND − χ²_KK) per point', color='#CCCCEE')
ax.set_title('Panel B: Per-point Δχ² by Acceleration Bin', color='#CCCCEE', fontsize=10)
ax.legend(fontsize=8, facecolor='#111122', labelcolor='#CCCCEE')
ax.text(1.0, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1] > 0 else 1, '+ve = KK better',
        color='#88FF88', fontsize=8, ha='center')

# ── Panel C: RAR coloured by class ───────────────────────────────────────────
ax = axes[1, 0]
for cls in classes:
    xs, ys = [], []
    for r in results:
        if r['class'] != cls:
            continue
        for gbar, gobs in zip(r['gbar_pts'], r['gobs_pts']):
            if gbar > 0 and gobs > 0:
                xs.append(np.log10(gbar))
                ys.append(np.log10(gobs))
    ax.scatter(xs, ys, s=1.2, alpha=0.25, color=colors[cls], label=labels[cls])

# Overlay model lines
g_bar_line = np.logspace(-13, -9, 200)
y_line     = g_bar_line / A0
ax.plot(np.log10(g_bar_line), np.log10(nu_KK(y_line)   * g_bar_line), '-',
        color='#4ECDC4', lw=2, label='KK')
ax.plot(np.log10(g_bar_line), np.log10(nu_MOND(y_line)  * g_bar_line), '--',
        color='#FF6B35', lw=1.5, label='MOND')
ax.plot(np.log10(g_bar_line), np.log10(g_bar_line), ':',
        color='#888888', lw=1, label='Newton')
ax.set_xlabel('log₁₀(g_bar) [m/s²]', color='#CCCCEE')
ax.set_ylabel('log₁₀(g_obs) [m/s²]', color='#CCCCEE')
ax.set_title('Panel C: RAR Coloured by Surface Brightness', color='#CCCCEE', fontsize=10)
ax.legend(fontsize=7, facecolor='#111122', labelcolor='#CCCCEE', markerscale=4)

# ── Panel D: Cumulative Δχ² vs y cutoff ─────────────────────────────────────
ax = axes[1, 1]
y_cuts = np.logspace(-1.5, 1.5, 60)

for cls in classes:
    mask_cls = all_cls_arr == cls
    y_c   = all_y[mask_cls]
    dc_c  = all_dchi2[mask_cls]
    cum_dchi2 = []
    for yc in y_cuts:
        bin_mask = y_c < yc
        if bin_mask.sum() > 0:
            cum_dchi2.append(np.sum(dc_c[bin_mask]))
        else:
            cum_dchi2.append(0)
    cum_dchi2 = np.array(cum_dchi2)
    ax.plot(y_cuts, cum_dchi2, '-', color=colors[cls], label=labels[cls], lw=1.5)

ax.axhline(0, color='#555577', ls='--', lw=1)
ax.axvline(0.5, color='#888888', ls=':', lw=1)
ax.axvline(3.0, color='#888888', ls=':', lw=1)
ax.text(1.2, ax.get_ylim()[0]*0.1 if ax.get_ylim()[0] < 0 else 0.1,
        'y=0.5–3\ntransition', color='#888888', fontsize=7, ha='center')
ax.set_xscale('log')
ax.set_xlabel('y cutoff (points with y < cutoff)', color='#CCCCEE')
ax.set_ylabel('Cumulative Σ(χ²_MOND − χ²_KK)', color='#CCCCEE')
ax.set_title('Panel D: Cumulative KK Advantage vs y Threshold', color='#CCCCEE', fontsize=10)
ax.legend(fontsize=8, facecolor='#111122', labelcolor='#CCCCEE')

plt.suptitle('KK vs MOND: Radial Acceleration Relation by Surface Brightness Class\n'
             f'(a₀ = cH₀/2π = {A0:.3e} m/s², fixed; 1 free param: M/L disk)',
             color='#EEEEFF', fontsize=11, y=1.01)

plt.tight_layout()
plt.savefig('plots/kk_rar_morphology.png', dpi=150, bbox_inches='tight',
            facecolor='#0A0A1A')
print("\nSaved: plots/kk_rar_morphology.png")

# ─── Summary statistics ────────────────────────────────────────────────────────
print("\n=== Full summary ===")
total_dchi2_by_class = {}
for cls in classes:
    mask_cls = all_cls_arr == cls
    total = np.sum(all_dchi2[mask_cls])
    n_pts = mask_cls.sum()
    total_dchi2_by_class[cls] = total
    print(f"  {cls}: total Δχ² = {total:+.1f}  over {n_pts} points  "
          f"(mean/pt = {total/max(n_pts,1):+.4f})")

# Where does most of KK advantage come from?
print("\n=== Advantage by y-range (all galaxies) ===")
ranges = [(0, 0.5, 'Deep MOND y<0.5'),
          (0.5, 3.0, 'Transition y=0.5-3'),
          (3.0, 100, 'Newtonian y>3')]
for ylo, yhi, label in ranges:
    mask = (all_y >= ylo) & (all_y < yhi)
    total = np.sum(all_dchi2[mask])
    n     = mask.sum()
    print(f"  {label:25s}: Δχ²={total:+8.1f}  n={n:4d}  mean/pt={total/max(n,1):+.4f}")

# Transition zone by class
print("\n=== Transition zone (y=0.5-3) Δχ² by class ===")
for cls in classes:
    mask = (all_cls_arr == cls) & (all_y >= 0.5) & (all_y < 3.0)
    total = np.sum(all_dchi2[mask])
    n     = mask.sum()
    print(f"  {cls}: Δχ²={total:+8.1f}  n={n:4d}  mean/pt={total/max(n,1):+.4f}")
