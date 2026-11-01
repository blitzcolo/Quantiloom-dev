#!/usr/bin/env python3
"""Diff two Vulkan api_dump logs after normalisation (P0.3 command-stream proof).

Usage: apidump_diff.py OLD.txt NEW.txt

Keeps the calls that define a compute migration's command stream (vkCmd*,
descriptor updates and allocations, and the creation of descriptor layouts, pipeline
layouts, pools, compute pipelines, buffers and images) together with their indented
parameters. Every hex handle is renumbered by first appearance, so two runs that only
differ in addresses compare equal. vkDestroy*, vkQueueSubmit and timing lines are
excluded on purpose.

Exit status: 0 when the normalised streams are equal, 1 when they differ, 2 on a usage
or input error. Output is ASCII.
"""

import difflib
import re
import sys

KEPT = re.compile(
    r"^(vkCmd\w+|vkUpdateDescriptorSets|vkAllocateDescriptorSets|vkResetDescriptorPool|"
    r"vkCreateDescriptorSetLayout|vkCreatePipelineLayout|vkCreateDescriptorPool|"
    r"vkCreateComputePipelines|vkCreateBuffer|vkCreateImage|vkCreateImageView)\b"
)
CALL = re.compile(r"^(vk\w+)\b")
HEX = re.compile(r"0x[0-9a-fA-F]+")


def normalise(path):
    handles = {}

    def renumber(match):
        token = match.group(0).lower()
        if int(token, 16) == 0:
            return "0x0"
        if token not in handles:
            handles[token] = "H%d" % len(handles)
        return handles[token]

    lines = []
    keeping = False
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            if not line[0].isspace():
                call = CALL.match(line)
                keeping = bool(call and KEPT.match(call.group(1)))
                if keeping:
                    lines.append(HEX.sub(renumber, line))
                continue
            if keeping:
                lines.append(HEX.sub(renumber, line))
    return lines


def main(argv):
    if len(argv) != 3:
        sys.stderr.write(__doc__)
        return 2
    try:
        old = normalise(argv[1])
        new = normalise(argv[2])
    except OSError as error:
        sys.stderr.write("cannot read input: %s\n" % error)
        return 2
    diff = list(difflib.unified_diff(old, new, argv[1], argv[2], lineterm="", n=2))
    sys.stdout.write("kept %d lines (old), %d lines (new)\n" % (len(old), len(new)))
    if not diff:
        sys.stdout.write("command streams are identical\n")
        return 0
    for line in diff:
        sys.stdout.write(line.encode("ascii", "replace").decode("ascii") + "\n")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
