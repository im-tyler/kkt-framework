# KKT Geodesic Framework

Code and data repository for:

**KKT Geodesic Framework: Kaluza-Klein-T with a₀ = cH₀/(2π) — Theory and Observational Tests**  
Tyler (2026)


Disclaimer: Created with Opus 4.6 High, Validated with Codex 5.3 xHigh

Preprint: https://doi.org/10.5281/zenodo.18753081

## Contents

### scripts/
| Script | Description |
|--------|-------------|
| `kk_verify_all.py` | Verifies all mathematical identities and numerical constants in the paper (56 checks) |
| `sparc_tests_abc.py` | SPARC rotation curve fitting: KK vs MOND vs gauge field, shape/scale tests |
| `kk_rar_morphology.py` | RAR residuals by surface brightness class (LSB/MED/HSB) |
| `kk_solar_system.py` | Solar system constraints: Cassini, perihelion, LLR, binary pulsars |
| `kk_z_dependence.py` | Redshift evolution of a₀(z) = cH(z)/(2π) and BTFR predictions |
| `kk_z_btfr_data.py` | BTFR normalization vs redshift figure with observational data |
| `kk_dimensional_reduction.py` | Dimensional reduction and Hole B closure |
| `kk_dS_force_law.py` | De Sitter force law and beta-exponent analysis |

### plots/
The four figures appearing in the paper.

### data/
`rotation_curves.tsv` — SPARC database (Lelli et al. 2016, publicly available at http://astroweb.cwru.edu/SPARC/)

## Requirements
```
pip install numpy scipy matplotlib
```

## Running the verification
```bash
python scripts/kk_verify_all.py
```
Expected output: 56 PASS, minor warnings on conjectured quantities.

## Key result
The KKT quadrature formula v(y) = sqrt(1 + 1/y) with a0 = cH0/(2pi) outperforms simple MOND on 66% of SPARC galaxies (171 galaxies, >= 5 data points) with zero free parameters at the theory level.

## License
MIT
