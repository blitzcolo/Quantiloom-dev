#!/usr/bin/env bash
# Render a scene set and snapshot every file it produced, keyed by config name.
#
# Written for the B5 facade move, whose only check is "the render did not
# change". Renders are bit-exact reproducible here (two runs of gltf_pbr_test
# are byte-identical), so the check is `cmp`, not a comparison of means -- box
# filtering preserves the mean, which makes a blur invisible to one.
#
#   ./scripts/render-tests/capture_baseline.sh renders/before
#   ...refactor...
#   ./scripts/render-tests/capture_baseline.sh renders/after
#   diff -r renders/before renders/after
#
# The scene list covers every branch inside the span being moved: all the
# fused/RGB modes, the hyperspectral cube, the sensor chain on and off, the
# atmosphere baker, and both scene loaders. A path not rendered here is a path
# the move is not checked on.
#
# Outputs are collected by mtime rather than by name: two configs share an
# output filename, and the sensor and per-band dumps are named by the code, not
# the config. Nothing in the working tree is deleted -- the repo root holds
# renders somebody may still want.
set -uo pipefail
cd "$(dirname "$0")/../.."

DEST="${1:?usage: capture_baseline.sh <output-dir>}"
CLI="${CLI:-./build/src/app/Release/Quantiloom.exe}"
[ -x "$CLI" ] || { echo "no CLI at $CLI -- build first" >&2; exit 2; }

SCENES=(
    gltf_pbr_test           # vis_fused, sensor on, glTF
    cornell_box_vis         # vis_fused, sensor off
    cornell_box_mwir        # mwir_fused: scattering + self-emission
    cornell_box_swir        # swir_fused: nearly all scattering
    cornell_box_lwir_atmos  # lwir_fused + the atmosphere baker
    multispectral_test      # the hyperspectral cube, a separate renderer
    cube_gltf               # rgb, sensor on
    usd_spectral_test       # rgb, USD loader
)

OUTPUT_GLOBS=(-name '*.exr' -o -name '*.png' -o -name '*.img' -o -name '*.hdr'
              -o -name '*.raw' -o -name '*.bsq' -o -name '*.csv')

mkdir -p "$DEST"
status=0
stamp="$(mktemp)"

for scene in "${SCENES[@]}"; do
    cfg="assets/configs/${scene}.toml"
    [ -f "$cfg" ] || { echo "MISSING $cfg" >&2; status=1; continue; }

    out="$DEST/$scene"
    mkdir -p "$out"

    # Mark the moment the render starts; only files younger than this are its
    # output. Guards against a stale file from an earlier run being copied and
    # then compared against itself -- which is how a batch that never ran once
    # reported a pass in this repo.
    sleep 1
    : > "$stamp"

    printf '%-24s ' "$scene"
    "$CLI" "$cfg" > "$out/stdout.log" 2>&1
    rc=$?

    # The CLI's exit code is unreliable, so judge the run by a line only a
    # completed render emits.
    if ! grep -qiE 'saved|wrote' "$out/stdout.log"; then
        echo "NO OUTPUT LINE (rc=$rc) -- see $out/stdout.log"
        status=1
        continue
    fi

    n=0
    while IFS= read -r f; do
        cp -p "$f" "$out/" && n=$((n + 1))
    done < <(find . -maxdepth 1 -type f -newer "$stamp" \( "${OUTPUT_GLOBS[@]}" \))

    if [ "$n" -eq 0 ]; then
        echo "RENDERED BUT WROTE NOTHING"
        status=1
    else
        echo "ok, $n file(s)"
    fi
done

rm -f "$stamp"
echo
echo "snapshot in $DEST"
exit "$status"
