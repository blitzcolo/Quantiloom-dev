# Offline export records

The [fusion-ready rig exporter](fusion/README.md) adds native Brown and
fisheye projection, separate geometry/path truth, and export schema v2. Existing
v1 records remain readable. Rebuild SDK consumers after updating public camera
configuration types. See [export.v2.schema.json](export.v2.schema.json).

SDK 0.5.0 records and verifies **export integrity**. It publishes offline images with a versioned JSON sidecar, a TOML configuration snapshot, file hashes and embedded record references. It does **not** yet provide verified replay or dataset ground truth. `reproducibility_verified` and `replay.replayable` are false; a successful integrity check is not a reproducibility claim.

The export schema remains version 1. SDK 0.5.0 adds the separate `UIntImage` / UINT EXR API, reserves individual output files across concurrent sessions, and applies the current camera pose to ordinary and hyperspectral offline renders. Existing floating-point image interfaces remain available.

## Export and verify

Metadata defaults to enabled for CLI single, batch and sequence renders, Qt sequence renders, and the SDK hyperspectral path used by both frontends. Viewport screenshots are unchanged. Disable recording in the scene configuration with:

```toml
[dataset]
metadata = false
```

Qt preserves the SDK-owned `[dataset]` table when opening, saving and snapshotting a document. Export does not save over the open document. The hyperspectral dialog passes the document's asset base directory to the SDK.

A Qt temperature sweep captures the configured physical-camera products as well as radiance. Each temperature sample creates independent detector state, runs its configured warmup, and publishes the products and preview together. A timeline sequence instead retains device history across acquisitions, including skipped ticks and reused frames.

For `frame.exr`, the export includes `frame.metadata.json` and `frame.replay.toml`. A frame's sidecar enumerates only that transaction's managed files. Camera products carry their own calibration and acquisition timing. Reused camera frames retain the original frozen capture description. Hyperspectral records include output and traced wavelength lists, reconstruction method and actual backend, plus any reconstruction fallback.

```text
Quantiloom.exe dataset-verify output/frame.metadata.json
python scripts/dataset_verify.py output/frame.metadata.json --inspect-exr
```

The CLI prints JSON and returns 0 for intact exports, 1 for invalid exports, or 2 for incorrect arguments. Its `checks` field is `export_integrity`: version, completion state, artifact paths, unique identities, sizes and SHA-256 digests. It does not validate input assets or execution equivalence.

The independent Python script additionally checks PNG summaries and camera matrix consistency. `--inspect-exr` requires OpenEXR and numpy and checks EXR summaries, dimensions, channel storage types and finite pixel values. It uses no renderer or SDK code. The schema is [export.schema.json](export.schema.json). `example/scene.toml` and `example/plate.gltf` provide a small portable input; run the config from the core repository so deployment resources remain available.

## Publication and recovery

`dataset::ExportSession::Create()` reserves the requested output path, its metadata sidecar and its TOML snapshot. `StagingPath()` reserves each additional destination until session destruction, even if the caller never registers that file. `RegisterFile()` makes a successfully closed staged file part of the transaction; `WriteImage()` performs both steps for floating-point EXR and PNG products.

Reservations are directories next to the destination, named `<filename>.quantiloom-export.lock`. For example, `frame.exr` uses `frame.exr.quantiloom-export.lock`; its staging files are inside that directory. Two sessions can write disjoint file sets in the same directory. A shared destination, including one addressed from a different subdirectory, is rejected. A file destination also conflicts with a destination below it, such as `bands` versus `bands/channel.exr`. Staging failure preserves any previous complete export.

Publication first atomically replaces the sidecar with `publishing`, then replaces and hashes the managed artifacts, and finally atomically publishes `complete`. Failure after publication starts leaves `failed` or `publishing`, which readers must reject. Old unlisted files are not members of the new sample. Readers must validate the expected record ID and hashes instead of relying on file existence.

Normal destruction removes staging files and releases only the session's own reservations. A process crash can leave one or more reservation directories behind. To recover:

1. Ensure no writer for the affected output set remains active. Do not remove another writer's reservations.
2. Inspect the metadata sidecar and any staged files. `publishing` and `failed` records are not usable samples. A previous `complete` record still requires hash verification.
3. Remove only the confirmed stale reservation directories for that set. A legacy `.quantiloom-export.lock` directory also blocks writes until inspected and removed.
4. Render the affected sample again. Removing reservations does not repair a partial export.

Failure to flush or close ENVI and TIFF writers is propagated to the session.

## Calibration conventions

Each product stores its actual native grid. Camera axes are right, down, forward; image coordinates start at the upper left and pixel centres are `(x+0.5, y+0.5)`. Transform arrays are row major and named by conversion direction. Translations use renderer world units, with an explicit conversion to metres. Orthographic products contain film dimensions and no perspective intrinsics.

Camera capture records are frozen before the renderer restores its state. They include first-row exposure midpoint, effective exposure interval, row timing and hashes of incoming and outgoing acquisition state. Radiance records retain the random seeds actually used by the frame. These descriptions do not identify a unique motion field for an exposure-integrated image.

## Integer EXR API

Use `ImageIO::WriteUIntEXR()` and `ImageIO::ReadUIntEXR()` for integer-valued
rasters. `UIntImage` owns a `Vector<u32>` and does not pass through the
floating-point `Image` container. A floating-point conversion can change IDs
above `2^24`, even when the resulting EXR is lossless.

```cpp
#include <io/ImageIO.hpp>

quantiloom::UIntImage ids;
ids.width = 3;
ids.height = 2;
ids.pixels = {0, 1, 16777217u, 2147483649u, 4294967295u, 17};
ids.metadata["source"] = "caller-provided identifiers";
auto written = quantiloom::ImageIO::WriteUIntEXR("ids.exr", ids);
if (!written) {
    // Report written.error(); do not publish this file as a successful output.
    return;
}
auto loaded = quantiloom::ImageIO::ReadUIntEXR("ids.exr");
// Check loaded before reading loaded.value().pixels.
```

The writer emits one `instance_id` channel of type UINT with ZIP compression,
unit sampling and a zero-origin native grid. The reader requires the same
channel name, type and sampling, and identical data/display windows starting
at `(0, 0)`. It accepts lossless NO, RLE, ZIPS, ZIP and PIZ compression and
rejects floating-point channels rather than converting them. Both methods
return an error for invalid input or I/O failure. String metadata is preserved;
reserved EXR header attributes cannot be replaced through the metadata map.

This interface stores caller-provided integers. It does not assign node
identities, produce masks from rendering, or establish that a pixel is valid
training evidence. It is a separate I/O API; `ExportSession::WriteImage()`
continues to accept the floating-point `Image` type.

## Compatibility and scope

`dataset-verify` checks exported file consistency, product references and
camera geometry records. It does not verify instance labels, semantic classes,
visibility, bounding boxes, mask areas, or correspondence between products.
Automatic training labels and a Qt annotation overlay viewer are not provided
by this version.

`reproducibility_verified` and `replay.replayable` remain false. In particular,
a sequence frame's TOML snapshot does not contain the complete acquisition
history or copied input resources required for independent replay.

The Qt offline-export integration requires SDK 0.5.0 or newer.
After reinstalling the SDK, run the Qt repository's `./build_wsl.sh` to
reconfigure and rebuild its consumers. A plain incremental link can copy a new
DLL without regenerating `SdkStamp.hpp` when installed files retain older
timestamps. The DLL next to the Qt executable, the installed SDK DLL, and the
SHA-256 recorded in the generated stamp must agree.
