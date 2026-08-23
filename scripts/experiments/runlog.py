#!/usr/bin/env python3
"""Run the renderer and write down what was run.

A number in a paper is only reproducible if the run behind it can be
reconstructed a year later, and the renderer records almost none of what that
takes: not the commit, not the resolved config, not the wall clock, not the
machine.  This wraps a CLI invocation and writes a manifest beside its outputs.

Deliberately a wrapper rather than a `--manifest` flag in the CLI.  The batch
mode layers a config, a shared override and a per-line override before it
renders, so the thing worth recording is the file as it existed plus the
overrides as given -- which the caller has and the CLI would have to
reconstruct.  A wrapper also costs the renderer nothing and cannot regress it.

Usage, as a library:

    from runlog import Run
    with Run("e6_furnace", out_dir) as run:
        run.render("assets/configs/furnace_lwir_e1.toml",
                   overrides={"renderer.spp": 256})

Usage, from a shell:

    runlog.py --out-dir evidence/e6 --label furnace \\
        assets/configs/furnace_lwir_e1.toml
"""

import argparse
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CLI = REPO / "build" / "src" / "app" / "Release" / "Quantiloom.exe"

# The project's two seed roots. Every experiment either uses these or derives
# from them, so that a number in the paper traces to one of two constants
# rather than to whatever was in the config that day.
SEED_SAMPLING = 0x547C
SEED_SENSOR = 0x548C


def git_state(repo):
    def raw(*args):
        """Bytes, always. A diff is not text: it can carry any encoding the
        repository contains, and decoding it under the console's code page
        turns a working call into a silent None on the first file whose name
        or contents the code page cannot represent."""
        try:
            result = subprocess.run(["git", "-C", str(repo), *args],
                                    capture_output=True, timeout=60)
            return result.stdout if result.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    def text(*args):
        out = raw(*args)
        return out.decode("utf-8", "replace").strip() if out is not None else None

    head = text("rev-parse", "HEAD")
    if head is None:
        return None
    return {
        "commit": head,
        "branch": text("rev-parse", "--abbrev-ref", "HEAD"),
        # The diff's hash, not just a dirty flag. An experiment run against
        # uncommitted work is the normal case while the work is in progress,
        # and "dirty: true" a year later does not say which uncommitted work.
        "dirty": bool(text("status", "--porcelain")),
        "diff_sha256": hashlib.sha256(raw("diff", "HEAD") or b"").hexdigest(),
    }


def sha256(path):
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Run:
    """One experiment's worth of renders, and the manifest describing them."""

    def __init__(self, label, out_dir, cli=DEFAULT_CLI, extra_repos=None):
        self.label = label
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cli = pathlib.Path(cli)
        self.records = []
        self.started = time.time()
        repos = {"Quantiloom": REPO}
        repos.update(extra_repos or {})
        self.manifest = {
            "label": label,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": platform.node(),
            "cli": str(self.cli),
            "cli_sha256": sha256(self.cli),
            "seeds": {"sampling": SEED_SAMPLING, "sensor": SEED_SENSOR},
            "repositories": {n: git_state(p) for n, p in repos.items()},
            "runs": self.records,
        }

    # ------------------------------------------------------------------
    def render(self, config, overrides=None, output=None, cwd=REPO, timeout=7200,
               expect_success=True):
        """Render one config, optionally with dotted-key overrides.

        Overrides go through a one-line batch manifest rather than by editing
        the config, because that is the only path the CLI offers that can set
        renderer.output per job -- and because it leaves the config on disk
        exactly as the manifest records it.
        """
        config = pathlib.Path(config)
        overrides = dict(overrides or {})
        if output is not None:
            # Absolute, always. A relative renderer.output in a batch override
            # is resolved beside the CONFIG, not beside the working directory --
            # so `renders/x.exr` from a config in assets/configs lands in
            # assets/configs/renders/, which is both surprising and somewhere
            # nobody looks. Naming the full path removes the question.
            resolved = pathlib.Path(output)
            if not resolved.is_absolute():
                resolved = pathlib.Path(cwd) / resolved
            overrides["renderer.output"] = resolved.as_posix()

        record = {
            "config": str(config),
            "config_sha256": sha256(config if config.is_absolute() else cwd / config),
            "overrides": overrides,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        if overrides:
            tokens = " ".join(f"{k}={_toml_value(v)}" for k, v in overrides.items())
            manifest_path = self.out_dir / f"_batch_{len(self.records):04d}.txt"
            manifest_path.write_text(
                f"{_as_posix(config, cwd)} | {tokens}\n", encoding="utf-8")
            command = [str(self.cli), "batch", str(manifest_path)]
            record["batch_manifest"] = str(manifest_path)
        else:
            command = [str(self.cli), str(config)]

        start = time.perf_counter()
        result = subprocess.run(command, cwd=str(cwd), capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=timeout)
        elapsed = time.perf_counter() - start

        log_path = self.out_dir / f"_log_{len(self.records):04d}.txt"
        log_path.write_text((result.stdout or "") + (result.stderr or ""),
                            encoding="utf-8")

        record.update({
            "command": command,
            "returncode": result.returncode,
            "seconds": round(elapsed, 3),
            "log": str(log_path),
        })
        if output is not None:
            resolved = pathlib.Path(output)
            resolved = resolved if resolved.is_absolute() else cwd / resolved
            record["output"] = str(output)
            record["output_sha256"] = sha256(resolved)

        self.records.append(record)

        if expect_success and result.returncode != 0:
            raise RuntimeError(
                f"{config} failed with code {result.returncode}; see {log_path}")
        return record

    # ------------------------------------------------------------------
    def note(self, **fields):
        """Attach anything else the experiment wants remembered."""
        self.manifest.setdefault("notes", {}).update(fields)

    def write(self):
        self.manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.manifest["total_seconds"] = round(time.time() - self.started, 3)
        path = self.out_dir / f"manifest_{self.label}.json"
        path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        return path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        path = self.write()
        print(f"manifest: {path}", file=sys.stderr)
        return False


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return '"' + str(value).replace("\\", "/") + '"'


def _as_posix(config, cwd):
    path = pathlib.Path(config)
    if not path.is_absolute():
        path = pathlib.Path(cwd) / path
    return path.as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", type=pathlib.Path)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--label", default="run")
    parser.add_argument("--cli", type=pathlib.Path, default=DEFAULT_CLI)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="dotted TOML override applied to every config")
    args = parser.parse_args()

    overrides = {}
    for entry in args.set:
        key, _, value = entry.partition("=")
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value

    with Run(args.label, args.out_dir, cli=args.cli) as run:
        for config in args.configs:
            record = run.render(config, overrides=overrides or None,
                                expect_success=False)
            status = "ok" if record["returncode"] == 0 else "FAILED"
            print(f"{status:>6}  {record['seconds']:>8.2f} s  {config}")


if __name__ == "__main__":
    main()
