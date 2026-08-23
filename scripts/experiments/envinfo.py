#!/usr/bin/env python3
"""Capture the machine an experiment ran on.

Every timing number and every render in the paper is a claim about a specific
GPU, driver and CPU, and none of that is recorded anywhere the renderer writes.
This collects it once into JSON, and prints a block that can be pasted into the
setup section rather than retyped from memory.

Nothing here is required for an experiment to run.  It is required for one to
be reported.

Usage:
    envinfo.py --out evidence/environment.json
    envinfo.py --markdown
"""

import argparse
import base64
import json
import pathlib
import platform
import re
import shutil
import subprocess
import sys


def run(command, timeout=20):
    """Run a command and return stdout, or None if it is not available.

    Decoded as UTF-8 rather than the console's code page: a localised Windows
    reports its edition in the local language, and on a Chinese install the
    default code page turns that into mojibake in a file the paper quotes.
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", errors="replace",
                                timeout=timeout, shell=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def powershell(script):
    """Run PowerShell and return its output as text, via base64.

    Setting [Console]::OutputEncoding does not survive redirection -- a piped
    PowerShell still writes in the console code page -- so a localised Windows
    edition name arrives as mojibake however the pipe is decoded. Encoding to
    base64 inside PowerShell and decoding here moves the string through as
    bytes, which no code page can reinterpret.
    """
    wrapped = ("[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("
               "[string](" + script + ")))")
    out = run(["powershell", "-NoProfile", "-Command", wrapped])
    if not out:
        return None
    try:
        return base64.b64decode(out).decode("utf-8").strip()
    except (ValueError, UnicodeDecodeError):
        return None


def nvidia():
    if shutil.which("nvidia-smi") is None:
        return None
    fields = "name,driver_version,memory.total,compute_cap,power.limit"
    out = run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"])
    if not out:
        return None
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append({
                "name": parts[0],
                "driver_version": parts[1],
                "memory_total": parts[2],
                "compute_capability": parts[3],
                "power_limit": parts[4] if len(parts) > 4 else None,
            })
    # The clock state a timing run was taken in. Not locked here -- locking is a
    # decision for the person running the experiment -- but recorded, because a
    # median that was taken while the card was thermally throttled is a
    # different number from one that was not.
    clocks = run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,"
                  "temperature.gpu,persistence_mode", "--format=csv,noheader"])
    return {"gpus": gpus, "clocks_at_capture": clocks}


def windows_hardware():
    if platform.system() != "Windows":
        return {}
    info = {}
    cpu = powershell("(Get-CimInstance Win32_Processor | "
                     "Select-Object -First 1 Name,NumberOfCores,"
                     "NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress)")
    if cpu:
        try:
            info["cpu"] = json.loads(cpu)
        except json.JSONDecodeError:
            info["cpu_raw"] = cpu
    memory = powershell("[math]::Round((Get-CimInstance Win32_ComputerSystem)."
                        "TotalPhysicalMemory / 1GB, 1)")
    if memory:
        info["memory_gb"] = memory
    os_name = powershell("(Get-CimInstance Win32_OperatingSystem).Caption + ' build ' + "
                         "(Get-CimInstance Win32_OperatingSystem).BuildNumber")
    if os_name:
        info["os"] = os_name
    # The English edition name too, so the paper does not have to quote a
    # localised one -- the build number is the part that identifies the OS.
    build = powershell("(Get-CimInstance Win32_OperatingSystem).Version")
    if build:
        info["os_version"] = build
    return info


def vulkan():
    if shutil.which("vulkaninfo") is None and shutil.which("vulkaninfoSDK") is None:
        return None
    out = run([shutil.which("vulkaninfo") or shutil.which("vulkaninfoSDK"), "--summary"],
              timeout=60)
    if not out:
        return None
    version = re.search(r"Vulkan Instance Version:\s*([\d.]+)", out)
    devices = re.findall(r"deviceName\s*=\s*(.+)", out)
    return {
        "instance_version": version.group(1) if version else None,
        "devices": [d.strip() for d in devices],
    }


def git_head(repo):
    head = run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    dirty = run(["git", "-C", str(repo), "status", "--porcelain"])
    if head is None:
        return None
    return {"commit": head, "dirty": bool(dirty), "path": str(repo)}


def collect(repos):
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "hardware": windows_hardware(),
        "nvidia": nvidia(),
        "vulkan": vulkan(),
        "repositories": {name: git_head(path) for name, path in repos.items()},
    }


def as_markdown(info):
    lines = ["### Environment", ""]
    hardware = info.get("hardware") or {}
    cpu = hardware.get("cpu") or {}
    if cpu:
        lines.append(f"- **CPU** {cpu.get('Name', '?').strip()} "
                     f"({cpu.get('NumberOfCores', '?')} cores / "
                     f"{cpu.get('NumberOfLogicalProcessors', '?')} threads)")
    if hardware.get("memory_gb"):
        lines.append(f"- **Memory** {hardware['memory_gb']} GB")
    if hardware.get("os"):
        lines.append(f"- **OS** {hardware['os']}")
    nv = info.get("nvidia") or {}
    for gpu in nv.get("gpus", []):
        lines.append(f"- **GPU** {gpu['name']}, driver {gpu['driver_version']}, "
                     f"{gpu['memory_total']}, compute {gpu['compute_capability']}")
    vk = info.get("vulkan") or {}
    if vk.get("instance_version"):
        lines.append(f"- **Vulkan** instance {vk['instance_version']}")
    for name, repo in (info.get("repositories") or {}).items():
        if repo:
            state = " (uncommitted changes)" if repo["dirty"] else ""
            lines.append(f"- **{name}** at `{repo['commit'][:12]}`{state}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--markdown", action="store_true",
                        help="print a block for the paper's setup section")
    parser.add_argument("--repo", action="append", default=[],
                        metavar="NAME=PATH",
                        help="extra repository to record the HEAD of")
    args = parser.parse_args()

    repos = {"Quantiloom": pathlib.Path(__file__).resolve().parents[2]}
    for entry in args.repo:
        name, _, path = entry.partition("=")
        repos[name] = pathlib.Path(path)

    info = collect(repos)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)

    print(as_markdown(info) if args.markdown else json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
