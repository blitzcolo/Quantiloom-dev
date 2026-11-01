---
name: physics-audit
description: Audit Quantiloom physics formulas (Planck blackbody, Wien, Stefan-Boltzmann, conductor Fresnel, GGX energy, Rayleigh/Mie/Henyey-Greenstein phase functions, Koschmieder visibility) against the stdlib-only Python reference harness in scripts/physics-audit. Use this when reviewing or changing any physics formula in shaders or C++ (src/shaders, spectral/material/atmosphere/sensor code), when the user asks whether a formula is physically correct, or when a physics unit test fails and you need an independent reference before touching tolerances.
---

# Physics Audit

`scripts/physics-audit/` holds the independent reference implementation used to audit formulas in shaders and C++. It deliberately uses only the Python standard library (CODATA 2018 constants) so it runs anywhere with `python3` — no venv, no pip.

## Layout

- `harness.py` — reference formulas: `planck_blackbody`, `wien_peak_wavelength`, `stefan_boltzmann_radiance`, `fresnel_exact_conductor` (matches `src/shaders/common.hlsli`), `fresnel_f0`, `koschmieder_visibility`, phase functions, GGX energy integral, Monte Carlo sphere integration.
- `test_harness.py` — spot checks that the harness itself is sane.
- `evidence/*.json` — audit findings per module (`{module, summary, findings}`): atmosphere, brdf, ir, spectral.

## Workflow

1. **Sanity-check the harness first:**
   ```bash
   cd /mnt/h/Quantiloom-dev/scripts/physics-audit && python3 test_harness.py
   ```
   Expect `All harness spot checks passed.`

   Careful: the shell working directory persists across commands. After working
   in `scripts/physics-audit/`, `cd` back to the repo root before running
   relative-path build/test commands (`cmake.exe --build build ...` will
   otherwise fail with "scripts/physics-audit/build is not a directory").

2. **Compare implementation vs reference.** Read the shader/C++ formula under audit, then evaluate the same inputs through the harness:
   ```bash
   python3 -c "
   import harness as H
   # Aluminum at 45 deg, n=0.27 k=3.29 — compare against shader output / unit test value
   print(H.fresnel_exact_conductor(0.7071, 0.27, 3.29))"
   ```
   The reference is ground truth. If implementation and harness disagree, the burden of proof is on the implementation — check units first (nm vs m, per-nm vs per-m radiance) before suspecting the math; unit mix-ups cause most "1e9 off" discrepancies.

3. **Record findings** in the matching `evidence/<module>_evidence.json`, following its existing `{module, summary, findings}` shape. An audit that isn't written down will be redone from scratch next month.

4. **Close the loop.** A confirmed formula bug gets: fix in shader/C++, a unit test in `tests/` pinning the correct value (see run-tests skill), and an end-to-end render check when the shader path is affected (see render-verify skill).

## Worked example: adjudicating a failing physics test

`AtmosphericConfigTest` failed pinning ClearDay `mie_beta_550nm = 2.0e-6` while the
implementation had `1.70e-4`. Adjudication took one harness call:

```bash
python3 -c "
import harness as H
for beta in (2.0e-6, 1.7e-4):
    print(beta, '->', H.koschmieder_visibility(beta), 'km')"
# 2e-06  -> 1956 km   (absurd for 'clear day')
# 1.7e-4 -> 23 km     (realistic)
```

Verdict: implementation correct (fixed per evidence finding ATM-01), test stale →
update the test, record the outcome (ATM-21). The point: the harness turns
"which side is right?" from a debate into a one-liner. Always check the evidence
JSONs for a prior finding before assuming a failure is a new bug.

## Extending the harness

New reference functions must stay stdlib-only, cite the formula source in the docstring (as existing functions do), and get a spot check in `test_harness.py`. Match the formulation used by the implementation being audited (note how `fresnel_exact_conductor` mirrors `common.hlsli`) so numeric comparison is direct, not "close enough".

## Known open physics debt (SRS)

Worth keeping in mind while auditing — these are documented, not new discoveries: NEE + BSDF sampling lacks MIS weights (FR-PT-03, double-counting risk); white furnace test missing (NFR-CORR-01); volume rendering shaders (Woodcock/HG) written but never called (FR-PT-07); noise seeding is non-reproducible (NFR-REPRO-01).
