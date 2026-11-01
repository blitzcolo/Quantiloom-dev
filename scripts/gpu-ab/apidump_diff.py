#!/usr/bin/env python3
"""Diff two Vulkan api_dump logs after normalisation (P0.3 command-stream proof).

Usage: apidump_diff.py [--host-sync] OLD.txt NEW.txt

Keeps the calls that define a migration's command stream (vkCmd*, descriptor updates
and allocations, and the creation of descriptor layouts, pipeline layouts, pools,
compute and ray tracing pipelines, acceleration structures and their build sizes,
samplers, query pools, events, command pools, pipeline caches, fences, buffers and
images) together with their indented parameters. Every hex handle is renumbered by
first appearance, so two runs that only differ in addresses compare equal.
vkDestroy*, vkQueueSubmit and timing lines are excluded on purpose, and so are
ray tracing group handles and stack sizes, which are driver-opaque.

--host-sync switches to the second view: the host's side of submission and
synchronisation (vkQueueSubmit, vkQueueWaitIdle, vkWaitForFences, vkResetFences,
vkGetFenceStatus, vkAllocateCommandBuffers, vkFreeCommandBuffers,
vkBeginCommandBuffer, vkEndCommandBuffer, vkResetCommandBuffer, vkDeviceWaitIdle),
anchored by the creation and destruction of command pools and fences so that a call
that moved relative to a pool's or fence's lifetime shows where it moved. Nothing
else is kept in that view.

Exit status: 0 when the normalised streams are equal, 1 when they differ, 2 on a usage
or input error. Output is ASCII.
"""

import difflib
import re
import sys

KEPT = re.compile(
    r"^(vkCmd\w+|vkUpdateDescriptorSets|vkAllocateDescriptorSets|vkResetDescriptorPool|"
    r"vkCreateDescriptorSetLayout|vkCreatePipelineLayout|vkCreateDescriptorPool|"
    r"vkCreateComputePipelines|vkCreateBuffer|vkCreateImage|vkCreateImageView|"
    r"vkCreateRayTracingPipelinesKHR|vkCreateAccelerationStructureKHR|"
    r"vkGetAccelerationStructureBuildSizesKHR|vkCreateSampler|vkCreateQueryPool|vkCreateEvent|"
    r"vkCreateCommandPool|vkCreatePipelineCache|vkCreateFence)\b"
)
HOST_SYNC = re.compile(
    r"^(vkQueueSubmit|vkQueueWaitIdle|vkWaitForFences|vkResetFences|vkGetFenceStatus|"
    r"vkAllocateCommandBuffers|vkFreeCommandBuffers|vkBeginCommandBuffer|vkEndCommandBuffer|"
    r"vkResetCommandBuffer|vkDeviceWaitIdle|"
    r"vkCreateCommandPool|vkDestroyCommandPool|vkCreateFence|vkDestroyFence)\b"
)
CALL = re.compile(r"^(vk\w+)\b")
HEX = re.compile(r"0x[0-9a-fA-F]+")


def normalise(path, kept):
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
                keeping = bool(call and kept.match(call.group(1)))
                if keeping:
                    lines.append(HEX.sub(renumber, line))
                continue
            if keeping:
                lines.append(HEX.sub(renumber, line))
    return lines


def main(argv):
    flags = [arg for arg in argv[1:] if arg.startswith("--")]
    paths = [arg for arg in argv[1:] if not arg.startswith("--")]
    if len(paths) != 2 or any(flag != "--host-sync" for flag in flags):
        sys.stderr.write(__doc__)
        return 2
    host_sync = "--host-sync" in flags
    kept = HOST_SYNC if host_sync else KEPT
    try:
        old = normalise(paths[0], kept)
        new = normalise(paths[1], kept)
    except OSError as error:
        sys.stderr.write("cannot read input: %s\n" % error)
        return 2
    diff = list(difflib.unified_diff(old, new, paths[0], paths[1], lineterm="", n=2))
    if host_sync:
        sys.stdout.write("host-sync view\n")
    sys.stdout.write("kept %d lines (old), %d lines (new)\n" % (len(old), len(new)))
    if not diff:
        sys.stdout.write("command streams are identical\n")
        return 0
    for line in diff:
        sys.stdout.write(line.encode("ascii", "replace").decode("ascii") + "\n")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
