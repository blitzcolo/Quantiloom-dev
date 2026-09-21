#pragma once

#include "core/Platform.hpp"
#include "core/Types.hpp"
#include "postprocess/CameraPipeline.hpp"

#include <vector>

namespace quantiloom::camera {

// The first-prelease hardware preset library (plan-0919 section 3). A preset
// is a complete, runnable CameraConfig: detector + optics + readout + ISP
// defaults. Hardware presets carry CalibrationStatus::HardwareReference and
// every non-public value is labelled in CameraPresetProvenanceNotes with one
// of the plan's grades: manufacturer (datasheet), digitized (from a maker
// graph), derived (computed from published values) or assumed. Two extra
// grades exist for the cases the plan calls out explicitly: "borrowed
// assumption" (a monochrome measurement reused for the colour variant, never
// claimed as a colour measurement) and "limit"/"estimated" for specification
// bounds and FAQ estimates. Missing hardware parameters are inherited from
// the matching generic family preset and listed as differences.
enum class CameraPresetKind : u32 {    // Generic closed-form families; every parameter is a published modelling
    // definition, for closed-form tests that need no external data.
    GenericCmos,             // Visible photon, Bayer colour, global shutter.
    GenericIngaas,           // SWIR photon, monochrome.
    GenericCooledPhotonIr,   // Cooled MWIR photon (InSb class), monochrome.
    GenericUncooledThermalIr,// LWIR VOx microbolometer, first-order thermal.
    // Hardware reference presets.
    Alvium1800U507Mono,
    Alvium1800U507Color,
    HamamatsuC12741_11Minus60C,
    HamamatsuC12741_11Minus70C,
    FlirA6751,
    Boson640Industrial,
};

// One labelled parameter value of a preset, for display in a host panel.
// `provenance` is one of: manufacturer / digitized / derived / assumed, or
// the explicit qualifiers "borrowed assumption", "limit" or "estimated".
struct PresetParameterNote {
    String parameter;  // CameraConfig path, e.g. "photon.read_noise_e_rms".
    String value;      // Human-readable value with unit.
    String provenance;
    String source;     // Datasheet URL, document id, or the assumption note.
};

/// Build the full camera config for a preset. The result always passes
/// ValidateCameraConfig. The returned config is enabled and uses
/// CameraInputKind::SpectralMeasurement; hosts override what they need.
[[nodiscard]] QL_API Result<CameraConfig, String>
MakePresetCameraConfig(CameraPresetKind kind);

/// English display name (Qt translates); stable across releases.
[[nodiscard]] QL_API String CameraPresetDisplayName(CameraPresetKind kind);

/// Stable lowercase snake_case token used by `[sensor] preset = "..."`.
[[nodiscard]] QL_API String CameraPresetToken(CameraPresetKind kind);

/// Parse a preset token; errors name the offending token for TOML messages.
[[nodiscard]] QL_API Result<CameraPresetKind, String>
CameraPresetFromToken(const String& token);

/// Every preset, in declaration order (generic families first).
[[nodiscard]] QL_API std::vector<CameraPresetKind> AllCameraPresets();

/// Per-parameter provenance for a preset: parameter path, value, grade and
/// source document/URL. Kept out of CameraConfig on purpose -- the config
/// body stays a pure runtime description (and must not grow ad-hoc metadata
/// fields); a panel that wants to show "manufacturer / digitized / assumed"
/// badges reads this list instead.
[[nodiscard]] QL_API std::vector<PresetParameterNote>
CameraPresetProvenanceNotes(CameraPresetKind kind);

} // namespace quantiloom::camera
