# src/libQuantiloom/

Modules: `core/` (config, log, types, spectral data), `scene/`, `renderer/` (Vulkan),
`io/` (glTF/USD/EXR/LUT loaders), `hs_core/` (hyperspectral), `atmos/` (NN
atmosphere), `postprocess/` (sensor chain).

Sources and internal headers live here; the public headers are in
`include/quantiloom/<module>/`, mirroring the same layout. Compiled once into
`quantiloom_core` (OBJECT) and linked two ways — see the root `CLAUDE.md` for which
targets take which.

## New code is internal by default

`QL_API` (defined in `core/Platform.hpp`) is the SDK's contract, not a convenience
marker. A symbol carrying it is exported from `Quantiloom.dll`, visible to
Quantiloom-Qt, and covered by SemVer — removing it later is a breaking change. 47
declarations have it; that number should grow only deliberately.

Write a new class here, in `src/libQuantiloom/`, with no `QL_API`. Two rules keep
that the default:

- **Needing a unit test is not a reason to export.** The tests link `quantiloom_core`
  and see every symbol regardless. This used to be the main way the export surface
  grew: 682 symbols, of which 411 had no consumer at all.
- **A new GUI capability extends the `ExternalRenderContext` facade** with a method
  plus a POD parameter struct. It does not export the class implementing the
  capability — the frontend needs to set a parameter, not to own an object.

Promoting something to public is three coupled steps, and the third is reviewed:

1. move the header to `include/quantiloom/<module>/` (`git mv`; the include strings
   do not change, only the search root),
2. add `QL_API`,
3. `./scripts/check_exports.sh --update` and commit `docs/abi/exports.golden`.

Skip step 1 and Quantiloom-Qt cannot include it. Skip step 3 and `./build_wsl.sh`
stops at the ABI gate before installing.

Prefer a free function over a class where it fits: `QL_API` on a class exports every
member, private ones included. Anything with private state belongs behind a pimpl —
private data in a public header puts `sizeof` into the ABI, which has silently broken
a Quantiloom-Qt build before (see `GenericSensor`).

## The consumer is the installed SDK, not this build tree

Quantiloom-Qt links `D:/Quantiloom-SDK/windows_amd64`. After changing public headers
or exported symbols, run `./build_wsl.sh` to reinstall — otherwise a correct core fix
appears to have no effect in the GUI.

## Portability

Windows is the primary target, but this code must stay buildable with MSVC, GCC, and
Clang (`std::filesystem` over `\\` paths, exact `#include` casing, no MSVC-only
constructs). The `build-and-install` skill has the full rules.

## Commits

**No Claude Code session link in a commit message.** No `Claude-Session:` trailer,
no `https://claude.ai/code/...` URL, in the subject, the body or a trailer. Same for
PR descriptions.
