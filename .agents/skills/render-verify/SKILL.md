---
name: render-verify
description: Verify renderer, shader, material, sensor, or atmosphere changes by actually rendering a scene with the Quantiloom CLI and inspecting the output. Use this after any change that affects rendered results (HLSL shaders, path tracing, spectral/material code, sensor chain, atmosphere, output encoding) before claiming it works — unit tests alone don't exercise the GPU pipeline end-to-end. Also use when the user asks to render a scene, test a TOML config, or reproduce a rendering issue.
---

# Render Verify (Quantiloom CLI)

The CLI renders one TOML config headlessly: `Quantiloom <config.toml>`. The TOML is the single source of truth — there are no CLI override flags.

## Standard verification run

```bash
cd /mnt/h/Quantiloom-dev   # MUST run from repo root: configs use relative asset paths,
                           # outputs land relative to CWD
./build/src/app/Release/Quantiloom.exe assets/configs/gltf_pbr_test.toml
```

`gltf_pbr_test.toml` is the default choice: it exercises glTF loading, textured PBR
materials, and the sensor chain, and emits all three inspectable outputs —
`gltf_pbr_output.exr`, `gltf_pbr_output.png` (Read tool can view it directly), and
`gltf_pbr_output_rawdn.exr` (sensor DN). A correct render shows the SciFiHelmet
model with metallic highlights and emissive details; horizontal banding is the
configured sensor FPN/row-noise simulation, not an artifact. Renders in seconds.

Use `cornell_box_vis.toml` instead when the glTF-Sample-Assets submodule is
missing -- its geometry and spectral data are all committed in-repo -- or a
config from the table below when your change targets a specific subsystem.

Prerequisites: a current build (see build-and-install skill). **If HLSL changed, recompile shaders first** — running the old `.spv` silently "verifies" nothing:

```bash
cmd.exe /c "src\shaders\compile_shaders.bat" </dev/null
cmake.exe --build build --config Release -j
```

Requires an NVIDIA RTX GPU with Vulkan RT on the Windows side.

## Picking a scene

Configs in `assets/configs/`, ordered by cost and coverage:

| Config | Exercises | Notes |
|---|---|---|
| `gltf_pbr_test.toml` | glTF + textured PBR + sensor chain, RGB mode | **Default** — EXR/PNG/rawdn outputs |
| `cornell_box_vis.toml` | Spectral path tracing, VIS band, ECOSTRESS materials | No submodule needed — fallback smoke test, ~1 s |
| `cube_gltf.toml` / `cube_usdc.toml` | glTF / USD loading | USD needs USD-enabled build |
| `metal_spheres_test.toml` | Conductor Fresnel (n,k) | Use for Fresnel/BRDF changes |
| `multispectral_test.toml` | Hyperspectral cube, multi-pass | Slow; ENVI output |
| `usd_spectral_test.toml` | USD + spectral curves + sensor DN | Broadest single config |
| `avocado.toml` | Textured glTF asset | |

Pick the cheapest config that exercises the changed code path; add `multispectral_test` only when spectral reconstruction or ENVI output changed.

## Checking the result

1. **Exit code non-zero = failure.** Read the log; the renderer uses spdlog and prints Vulkan/asset diagnostics.
2. **Output freshness:** `ls -l --time-style=full-iso <output>` — the `output` path from the config's `[renderer]` section (e.g. `cornell_box_vis.exr`). Stale timestamp means the render silently didn't write.
3. **Visual check:** configs that emit `.png` previews can be inspected directly with the Read tool. All-black, all-white, or NaN-speckled output is a failure even with exit code 0.
4. **Numeric check (EXR/ENVI):** compare against the previous output — render once on the base commit, once with the change, and diff statistics. Expected-unchanged code paths must produce statistically identical images (rendering is currently seeded from `std::random_device`, so bit-exact repeats are NOT possible — compare means/histograms, not bytes; see SRS NFR-REPRO-01).
5. Side outputs matter: `_rawdn.exr` (sensor DN), `_bands/` per-band dumps, `.hdr/.dat` ENVI pairs — check the ones your change touches.

## Reporting

State which config was rendered, the exit code, and what was actually inspected (image looked at, statistics compared). "It built" is not verification.
