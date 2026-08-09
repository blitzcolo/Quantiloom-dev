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

## Two render gates, and what each is blind to

Neither is `ctest`; both need a GPU and both run from `build_wsl.sh`.

| Gate | Asks | Blind to |
|---|---|---|
| `run_furnace_suite.sh` | what a surface does with light once it arrives | anything about how it arrives — no sun, no sky, no scene outside the cavity |
| `run_illumination_suite.sh` | how light reaches a surface: occlusion, open-sky exactness, indirect | radiometry of the surface itself |

Two checkers, `check_dispersion.py` and `check_hero_wavelength.py`, are **red on
`main` and were red before the bounce work** — same figures to four decimals at
`b09706a`. Don't read them as a regression you caused; do fix them if you are in
VIS_FUSED's dispersion path.

## Commits

**No Claude Code session link in a commit message.** No `Claude-Session:` trailer,
no `https://claude.ai/code/...` URL, in the subject, the body or a trailer. Same for
PR descriptions.
