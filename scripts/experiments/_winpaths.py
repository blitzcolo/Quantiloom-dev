"""Refuse to run when this script's output paths are not paths.

These scripts drive the Windows CLI and write into the manuscript tree, so they
hardcode Windows paths:

    FIGURES = pathlib.Path(r"H:\\quantiloom-paper\\figures")

Run under WSL's interpreter that is not a path at all. pathlib takes the whole
string as a single directory NAME, and the first `mkdir(parents=True)` or
`savefig` creates it inside the current working directory -- which is the
repository -- with the colon and backslashes mapped into the Unicode private use
area, so that afterwards not even a shell glob will match it. The script then
prints "wrote H:\\quantiloom-paper\\figures/fig5_sensor_chain.png" and exits 0
while the real figure sits untouched.

That is the worst failure shape available: it looks like success. It happened
once, and a stale Fig. 5 got as far as being presented as a regenerated one
before the junk directory in `git status` gave it away.

Usage, immediately after the path constants:

    from _winpaths import require_windows_paths
    require_windows_paths(EVIDENCE, FIGURES)
"""
import pathlib
import sys


def require_windows_paths(*paths: pathlib.Path) -> None:
    """Exit with an explanation if any path has no drive letter."""
    driveless = [p for p in paths if not p.drive]
    if not driveless:
        return
    name = pathlib.Path(sys.argv[0]).name or "this script"
    raise SystemExit(
        f"{name} writes to Windows paths and must run under the Windows "
        f"interpreter, not WSL's.\n"
        f"  Driveless here: {', '.join(repr(str(p)) for p in driveless)}\n"
        f"  Under WSL each of those becomes a directory of that literal name "
        f"inside the repository, and the real output is never written.\n"
        f"  Use the Windows python (the paper-exp venv has numpy, OpenEXR and "
        f"matplotlib)."
    )
