# src/shaders/

HLSL compiled to SPIR-V by the Windows DXC (Vulkan SDK). `.spv` files are build
outputs and are gitignored — never commit or hand-edit them.

## After editing any shader

```bash
cmd.exe /c "src\shaders\compile_shaders.bat" </dev/null   # from repo root, ~1 s
```

A C++ rebuild alone does **not** pick up shader edits. The `</dev/null` is required:
the batch file calls `pause` on every error path and would hang a non-interactive
run there.

## Adding a shader

`compile_shaders.bat` is a hardcoded sequence of 13 invocations with literal
`[n/13]` labels — not a glob. A new shader must be added there (and the counts
renumbered) or it is silently never compiled.

## Stage extensions

`.rgen` / `.rchit` / `.rmiss` ray tracing stages, `.comp` and `.comp.hlsl` compute,
`.hlsli` shared headers (included, never compiled standalone).

## Indirect light is a residual, not a replacement

Every spectral band keeps its analytic ambient terms — the unoccluded uniform sky
dome, and the split-sum IBL against it — and `TraceEnvBounceResidual` traces one
ray per hit carrying only the **difference** between what the hemisphere actually
sends back and what those terms assumed:

```
Corr = rho(λ_b) · W_dir · ( L_in(λ_b) − L_base(λ_b) )
```

Three consequences worth knowing before touching any of it:

- **An open scene is unchanged, exactly.** Every bounce ray escapes and the miss
  shader returns the base it is subtracted from, so `Corr` is zero to the bit —
  verified in `check_sky_equiv.py`, and the reason that check has no tolerance
  for Monte Carlo noise. Deleting an analytic term and letting the ray carry the
  whole integral would be the same in expectation and far noisier.
- **An isothermal cavity is exact at 1 spp**, by the same argument run backwards:
  the incoming radiance *is* the base, so the furnace gate keeps working.
- **Russian roulette is unbiased without a closure.** Killing a path leaves the
  analytic term standing, so the fallback is the sky rather than zero. Depth caps
  behave the same way.

Both factors are evaluated at the **same** sampled wavelength, carried in
`Payload::heroLambda`. That correlation is the point of the ray — `⟨ρ⟩⟨L⟩` is not
`⟨ρL⟩`, and a quartz cavity was 1.15% wrong when the bands sent a whole-band ray
and multiplied by a band average. One ray either way.

RGB mode has none of this and spawns no bounce ray from an opaque surface. It is
the interactive preview; leave it that way.

## An RGB colour has one spectral meaning, and only inside 380-780 nm

Reflectance and illuminant both go through Jakob & Hanika: a sigmoid of a
quadratic in wavelength, three coefficients read from the table on binding 25.
`core/RgbToSpectrum.hpp` has the fit; what matters when editing a shader is the
shape of the API and the two rules around it.

```hlsl
float4 s = FetchRgbSpectrum(rgbToSpectrumTable, rgb);   // once per colour
float  r = RgbSpectrumAt(s, lambda);                    // once per wavelength
```

**Fetch outside the wavelength loop.** The coefficients depend only on the
colour. VIS_FUSED reads up to five reflectances and three illuminants at each of
32 wavelengths; fetching inside the loop would be 256 table lookups and two
thousand buffer loads per hit. Hoisted, it is eight lookups and four flops per
wavelength — cheaper than the three `exp()` the old Gaussian mapping cost. Every
call site in `closesthit.rchit` and `miss.rmiss` is already arranged this way;
adding one inside a loop is the mistake to avoid.

**Grey is carried, not fitted.** The `.w` of a fetched spectrum is the
reflectance itself when the triple is achromatic, and negative otherwise. Every
dielectric without `KHR_materials_specular` has F0 = (0.04, 0.04, 0.04) and both
render gates use grey scenes, so this is what lets a change here be checked
against a bit-identical baseline rather than argued about.

**Outside the fitted band it says nothing, and saying nothing is the answer.**
The quadratic keeps growing past 780 nm and the sigmoid saturates — toward 1 for
a saturated warm colour, which in a thermal band is a mirror where a wall should
be. `RgbSpectrumAt` clamps lambda to the fit domain, but that is a guard rail,
not permission: NIR, SWIR, MWIR and LWIR fall back to the material's own
`ir_emissivity` through Kirchhoff, and `ResolveMaterialSpectra` warns at load
time which materials that applies to. `allowRgbUpsample = false` is the same
rule for sheen and diffuse transmission.

The predecessor failed in the opposite direction and hid it: past 900 nm the
Gaussian basis sum underflowed its own divide-by-zero guard, so a grey 0.5
surface came back as 0.09 at 900 nm and 0.00 at 1200. Nobody chose that, and it
is why NIR used to disagree with SWIR and MWIR about the same scene's geometry.

### An RGB light source means D65

An emitter is not bounded by 1, so it splits: the chroma is fitted as a
reflectance and the magnitude rides alongside.

```
L(lambda) = scale * s(c; lambda) * d65(lambda),   scale = 2 * max(rgb)
```

`d65` is the `.w` of the CIE buffer at binding 19, normalised so a spectrum
equal to it integrates to Y = 1 through the estimator in this directory.

The D65 factor is a correction, not a preference. A flat spectrum is the
equal-energy illuminant E, which is linear sRGB (1.205, 0.948, 0.909) — so white
in, warm out. That used to be patched by scaling the final radiance by
`chromaR_correction` and `chromaB_correction`, defaulting to E's own G/R and
G/B, on every VIS_FUSED render including the spectrally correct ones. Both
default to 1.0 now. If you find yourself reaching for them, the illuminant is
the thing to fix.

## Two render gates, and what each is blind to

Neither is `ctest`; both need a GPU and both run from `build_wsl.sh`.

| Gate | Asks | Blind to |
|---|---|---|
| `run_furnace_suite.sh` | what a surface does with light once it arrives | anything about how it arrives — no sun, no sky, no scene outside the cavity |
| `run_illumination_suite.sh` | how light reaches a surface: occlusion, open-sky exactness, indirect | radiometry of the surface itself |

A third pair, `check_dispersion.py` and `check_hero_wavelength.py`, is run by
neither gate and has to be invoked by hand. Both edit
`assets/models/prism_*.gltf` in place and restore it in a `finally`, so they
cannot run concurrently with each other or with anything else reading those
models.

Both spent six weeks red for a reason that was in no shader: three copies of
the prism models sat under `assets/configs/assets/models/`, and
`ResolveConfigPath` prefers the config's own directory, so every render read
the duplicates while the checkers edited the originals. A patched glTF that
nothing loads renders identically to the unpatched one — which
`check_dispersion` reported as "switching dispersion on changed nothing", and
`check_hero_wavelength` reported as a bias floor, because its reference and its
test case then differed only by a config-injected IOR that *did* take effect.
The duplicates are deleted and `ResolveConfigPath` now warns when a path
resolves two ways. If a checker ever again insists a shader change did nothing,
read the render log's `Loading glTF model:` line before believing it.

## Commits

**No Claude Code session link in a commit message.** No `Claude-Session:` trailer,
no `https://claude.ai/code/...` URL, in the subject, the body or a trailer. Same for
PR descriptions.
