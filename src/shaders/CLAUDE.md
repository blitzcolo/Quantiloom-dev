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
