# src/libQuantiloom/

Built as a SHARED library (`Quantiloom.dll`). Modules: `core/` (config, log, types,
spectral data), `scene/`, `renderer/` (Vulkan), `io/` (glTF/USD/EXR/LUT loaders),
`hs_core/` (hyperspectral), `atmos/` (NN atmosphere), `postprocess/` (sensor chain).

## Exporting public API

Every publicly consumed class, struct, or free function needs `QL_API` (defined in
`core/Platform.hpp`):

```cpp
class QL_API Config { ... };
```

Omitting it compiles cleanly here but fails to link in the tests and in
Quantiloom-Qt. ~83 declarations across the library already use it.

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
