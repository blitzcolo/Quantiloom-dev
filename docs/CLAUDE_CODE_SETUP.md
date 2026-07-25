# Claude Code setup for Quantiloom

What was configured, why, and what each developer has to install by hand.
Every command and number below was produced by running it, not by reading docs —
`README.md` in particular is stale and was not used as a source.

## What is in the repo

| Path | Purpose |
|---|---|
| `CLAUDE.md` | 54 lines. Repo map, the WSL2-shell/Windows-toolchain rule, build and test commands, conventions. Loads every session. |
| `tests/CLAUDE.md` | 33 lines. Loads when working under `tests/`. |
| `src/shaders/CLAUDE.md` | 25 lines. Loads when working under `src/shaders/`. |
| `src/libQuantiloom/CLAUDE.md` | 29 lines. Loads when working under `src/libQuantiloom/`. |
| `.claude/settings.json` | Read-deny rules, `clangd-lsp` plugin, three `PostToolUse` hooks. |
| `.claude/skills/` | 6 skills (4 pre-existing, 2 added). |
| `.clangd` | Points clangd at `build-cdb` and restores the MSVC system include paths. |
| `.clang-tidy` | Curated lint checks. clangd reads it automatically — no lint command, no CI step. |
| `scripts/gen_compile_commands.sh` | Regenerates the compilation database; `--check` reports staleness. |
| `scripts/hooks/` | The three hook scripts. |

`.gitignore` was narrowed so this configuration is shareable: it used to ignore
all of `.claude/`, which is why the four original skills survived only as
`git add -f`. It now ignores just `.claude/settings.local.json`,
`.claude/worktrees/`, root-level render outputs, `build-cdb/`, and
`vendor/bc7enc_rdo/`.

## Manual setup, per developer

**1. A `clangd` shim on PATH.** The repo builds with the Windows MSVC toolchain,
so code intelligence must run the *Windows* `clangd.exe`. A Linux clangd cannot
work here: WSL2 has no C++ toolchain at all (no `g++`, no `/usr/include/c++`), so
it fails on the first standard header with `'cstdint' file not found`.

`clangd.exe` cannot open WSL paths, so the shim adds `--path-mappings`:

```sh
mkdir -p ~/.local/bin
cat > ~/.local/bin/clangd <<'SH'
#!/bin/sh
exec "/mnt/c/Users/<you>/AppData/Local/Programs/CLion/bin/clang/win/x64/bin/clangd.exe" \
     --path-mappings=/mnt/c=C:/,/mnt/d=D:/ "$@"
SH
chmod +x ~/.local/bin/clangd
```

Both sides of a mapping must be absolute — `/mnt/c=C:` is rejected with
`Invalid -path-mappings: Path not absolute: C:`.

The `clangd-lsp` plugin invokes a bare `clangd`, and Claude Code's process PATH
does not include `~/.local/bin` (it is a non-login shell, so `~/.profile` is not
read). Symlink the shim into a directory that is already on PATH, or add
`~/.local/bin` to PATH in `~/.bashrc` and restart the terminal.

Use CLion's bundled clangd 23 over the Visual Studio one (clangd 20) — against
MSVC 14.50's STL it reports 19 errors on `VulkanContext.cpp` where clangd 20
reports 31.

**2. The compilation database.**

```sh
./scripts/gen_compile_commands.sh           # ~13 s
./scripts/gen_compile_commands.sh --check   # is it stale? exits 1 if so
```

Re-run it after adding source files or changing CMake. clangd falls back to
heuristic flags for a file it cannot find in the database and reports nothing,
so staleness has to be asked for. `--check` compares the database against
`git ls-files` and against every CMake file's mtime.

Inside Claude Code the `post-write-cdb-staleness.sh` hook asks automatically,
on exactly the edits that can invalidate the database.

Nothing else needs installing. The `OpenEXR` Python package and numpy/scipy/
scikit-learn (for the spectral baker) are already present.

## Before and after

There was no `CLAUDE.md` anywhere before this work, so "before" is not a smaller
number — it is nothing at all, with every session rediscovering the build setup
by trial and error.

| | Value |
|---|---|
| Root `CLAUDE.md` loaded every session | 2,358 chars, ~589 tokens |
| Subdirectory files (load only in that directory) | 3,243 chars total, ~809 tokens |
| 6 skill name+descriptions (always loaded) | 2,350 chars, ~587 tokens |
| Skill bodies | loaded only when selected |

Read surface, measured with `rg --files .` (a repo-wide walk honouring
`.gitignore`): 6,845 files visible, of which **6,518 (95.2%)** were third-party
submodule content — `assets/refractiveindex/` (4,076),
`assets/models/glTF-Sample-Assets/` (2,407), `docs/doxygen-awesome-css/` (35).
All three are now read-denied. Actual source is ~200 files.

Command facts now recorded instead of rediscovered:

| | |
|---|---|
| Full suite | 863 tests, 839 passed / 24 skipped / 0 failed, ~3 s |
| `--gtest_brief=1` | 122 lines of output instead of 2,086 |
| `--gtest_filter` single suite | 0.044 s vs 3 s |
| `ctest -R` | cannot select a case — one aggregate test only |
| Incremental C++ build | 76 s |
| Shader compile | 1 s |
| clangd on `Config.cpp` | 0 errors (5 before the include fix) |

## Acceptance

- Read-deny verified: reading `assets/refractiveindex/.../Babar.yml` is refused;
  `assets/spectral/material_summary_rii.csv`, `assets/configs/`, and
  `src/libQuantiloom/` remain readable.
- Shader hook verified firing on a live `Edit`: `raygen.spv` mtime advanced
  without anyone running the compiler.
- Test-registration hook verified against synthetic payloads: silent for a
  registered file, correct warning naming `TEST_CORE_SOURCES` for an
  unregistered one.
- LSP verified end to end: `documentSymbol` returns the full `Config` tree with
  resolved types; diagnostics stream (it flagged an unused include in
  `SpectralData.hpp`).
- Suite still green after all changes: 863 tests, exit 0.

Known wart: `findReferences` lists each file twice, once WSL-relative and once as
`D:/...`. The path mapping is not symmetric on the way back because
`compile_commands.json` stores native Windows paths. Results are correct;
deduplicate by basename.

## Linting

`.clang-tidy` is read by clangd automatically, so lint warnings arrive as LSP
diagnostics while editing. Verified on `io/ImageIO.cpp`: 8 items, tagged
`source: clang-tidy`, including
`bugprone-implicit-widening-of-multiplication-result` — a real bug class in
numeric code.

The check list was tuned by measurement, not taste. Across a six-file sample the
default `bugprone-*,performance-*,misc-*` produced 119 warnings; excluding two
checks brought that to 33, with most files at 0-4:

| Excluded | Why |
|---|---|
| `misc-const-correctness` | 86 of the 119. A style preference, not a defect. |
| `bugprone-easily-swappable-parameters` | 6 of 8 hits on `core/Config.cpp`, all correct code. |
| `misc-non-private-member-variables-in-classes` | The GPU-facing structs are deliberately plain data. |
| `misc-no-recursion` | Scene-graph and spectral traversal are recursive by design. |
| `misc-include-cleaner` | Duplicates clangd's own unused-include diagnostic. |

`app/main.cpp` still reports 28 — it is a 1,000-line CLI entry point full of
numeric conversions. Treat that as a backlog, not a regression.

For a batch run (the VS-bundled binary, which needs the system includes spelled
out because they live in vcvars' `INCLUDE`):

```sh
"/mnt/c/Program Files/Microsoft Visual Studio/18/Enterprise/VC/Tools/Llvm/bin/clang-tidy.exe" \
  -p build-cdb --quiet \
  '--extra-arg=-imsvcC:/Program Files (x86)/Windows Kits/10/include/10.0.26100.0/ucrt' \
  src/libQuantiloom/io/ImageIO.cpp
```

6-18 s per file, which is why this is not a `PostToolUse` hook: clangd already
streams the same diagnostics, and the shader hook it would sit beside costs 1 s.

No `clang-format`. The existing style is already consistent (no tabs anywhere;
braces attached in 130 places vs 15 detached), and reformatting would rewrite the
codebase: LLVM and Google styles both touch 145% of lines, and even Microsoft —
the closest fit — touches 52%, destroying `git blame` and colliding with any
in-flight branch.

## Deliberately not done

- **`worktree.sparsePaths`** — the root `CMakeLists.txt` unconditionally
  `add_subdirectory()`s all five `src/` subdirectories (lines 429-441), so no
  subset configures. The minimal closure is the whole repo. `assets/` (3.8 GB) is
  the only thing big enough to be worth excluding, and both the test suite and
  `render-verify` need it. (`symlinkDirectories` for the per-worktree 1.7 GB
  `.cpm_cache` remains the one idea worth revisiting.)
- **`pyright-lsp`** — Python is 12 files / 5.4% of the code.
- **HLSL code intelligence** — `clangd-lsp` maps none of
  `.hlsl/.hlsli/.rgen/.rchit/.rmiss/.comp`, and no shader LSP exists in the
  marketplace. `src/shaders/CLAUDE.md` spells out the registration points by hand
  as compensation.
- **A `PostToolUse` clangd check** — the plugin already streams diagnostics, and
  `clangd --check` counts suppressed header errors (18 on a clean
  `VulkanContext.cpp`), which would train everyone to ignore hook output.
- **A `Stop` hook proposing `CLAUDE.md` edits** — 139 lines total, freshly
  verified; not worth a transcript re-read every turn. Revisit if they grow.
- **`.claude/rules/`** — no convention repeats across three or more directories.

## Maintenance

- Review `CLAUDE.md` changes like any other code. Each line here traces to a
  command that was run; keep that bar.
- Re-run `./scripts/gen_compile_commands.sh` after adding files or changing CMake.
- After a major model release, re-read these files and delete anything that was
  written to work around an older model's limitations.
- For a change spanning several modules, have the agent write the plan to a
  markdown file in the repo before editing — long sessions compact, and a saved
  plan survives where conversation history may not.
- `.claude/settings.json` carries machine-specific absolute paths in
  `additionalDirectories` and `autoMode`. If a second developer joins, move those
  to `.claude/settings.local.json`.
