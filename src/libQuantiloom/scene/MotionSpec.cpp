#include "scene/MotionSpec.hpp"

#include "core/Log.hpp"

#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>

namespace quantiloom::scene {

namespace {

String Trim(StringView text) {
    usize begin = 0;
    usize end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
    return String(text.substr(begin, end - begin));
}

/// Reads three doubles out of an array-valued key. A short or missing array
/// leaves @p out alone, so a default survives.
bool ReadVec3(const Config& table, StringView key, glm::dvec3& out) {
    const auto values = table.GetDoubleArray(key);
    if (values.size() < 3) return false;
    out = glm::dvec3(values[0], values[1], values[2]);
    return true;
}

/// The three expression strings of one channel, as authored. Present only when
/// the array has three entries; a shorter one is a mistake worth reporting.
bool ReadExpr3(const Config& table, StringView key, String (&out)[3], const String& owner,
               Vector<String>& warnings) {
    if (!table.Has(key)) return false;
    const auto values = table.GetStringArray(key);
    if (values.size() != 3) {
        warnings.push_back(owner + ": motion segment `" + String(key) + "` needs three " +
                           "expressions, found " + std::to_string(values.size()) +
                           "; channel ignored");
        return false;
    }
    out[0] = values[0];
    out[1] = values[1];
    out[2] = values[2];
    return true;
}

}  // namespace

// ============================================================================
// Durations
// ============================================================================

Result<f64, String> ParseDurationString(StringView text) {
    const String trimmed = Trim(text);
    if (trimmed.empty()) return Result<f64, String>::Err("empty duration");

    usize consumed = 0;
    f64 value = 0.0;
    try {
        value = std::stod(trimmed, &consumed);
    } catch (const std::exception&) {
        return Result<f64, String>::Err("\"" + trimmed + "\" is not a number followed by a unit");
    }

    String unit = Trim(StringView(trimmed).substr(consumed));
    std::transform(unit.begin(), unit.end(), unit.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (unit.empty() || unit == "s" || unit == "sec" || unit == "secs" || unit == "second" ||
        unit == "seconds") {
        return Result<f64, String>(value);
    }
    if (unit == "ms") return Result<f64, String>(value * 1e-3);
    if (unit == "min" || unit == "mins" || unit == "minute" || unit == "minutes") {
        return Result<f64, String>(value * 60.0);
    }
    if (unit == "h" || unit == "hr" || unit == "hrs" || unit == "hour" || unit == "hours") {
        return Result<f64, String>(value * 3600.0);
    }
    if (unit == "d" || unit == "day" || unit == "days") {
        return Result<f64, String>(value * 86400.0);
    }
    return Result<f64, String>::Err("unknown time unit \"" + unit + "\" in \"" + trimmed +
                                    "\" (use s, ms, min, h or d)");
}

std::optional<f64> ReadDuration(const Config& table, StringView key, const String& owner,
                                Vector<String>& warnings) {
    if (!table.Has(key)) return std::nullopt;

    // A bare number is the common case and needs no string round trip; only a
    // quoted value goes through the unit grammar.
    const String text = table.GetString(key);
    if (text.empty()) {
        return table.GetDouble(key);
    }
    auto parsed = ParseDurationString(text);
    if (!parsed) {
        warnings.push_back(owner + "." + String(key) + ": " + parsed.error());
        return std::nullopt;
    }
    return *parsed;
}

// ============================================================================
// Keyframe CSV
// ============================================================================

Result<Vector<MotionKey>, String> LoadMotionKeysCsv(const String& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return Result<Vector<MotionKey>, String>::Err("cannot open keyframe file \"" + path + "\"");
    }

    Vector<MotionKey> keys;
    String line;
    u32 lineNumber = 0;
    u32 malformed = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        const String trimmed = Trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;

        Vector<f64> fields;
        std::istringstream stream(trimmed);
        String cell;
        bool numeric = true;
        while (std::getline(stream, cell, ',')) {
            const String value = Trim(cell);
            try {
                fields.push_back(std::stod(value));
            } catch (const std::exception&) {
                numeric = false;
                break;
            }
        }
        if (!numeric) {
            // A header row is normal on the first line and a mistake anywhere
            // else, and the count at the end says which happened.
            if (lineNumber > 1) ++malformed;
            continue;
        }
        if (fields.size() < 4) {
            ++malformed;
            continue;
        }

        MotionKey key;
        key.t_s = fields[0];
        key.hasPosition = true;
        key.position = glm::dvec3(fields[1], fields[2], fields[3]);
        if (fields.size() >= 8) {
            key.hasRotation = true;
            key.rotation = glm::normalize(glm::dquat(fields[7], fields[4], fields[5], fields[6]));
        }
        keys.push_back(key);
    }

    if (keys.empty()) {
        return Result<Vector<MotionKey>, String>::Err("keyframe file \"" + path +
                                                      "\" holds no usable rows");
    }
    std::stable_sort(keys.begin(), keys.end(),
                     [](const MotionKey& a, const MotionKey& b) { return a.t_s < b.t_s; });
    if (malformed > 0) {
        QL_LOG_WARN("Motion keyframes: {} row(s) of \"{}\" were not four or more numbers "
                    "and were skipped",
                    malformed, path);
    }
    return Result<Vector<MotionKey>, String>(std::move(keys));
}

// ============================================================================
// The motion table
// ============================================================================

namespace {

void ParseKeys(const Config& table, const String& owner, const MotionPathResolver& resolvePath,
               MotionSpec& spec, Vector<String>& warnings) {
    const String keysFile = table.GetString("keys_file");
    if (!keysFile.empty()) {
        const String resolved = resolvePath(keysFile);
        auto loaded = LoadMotionKeysCsv(resolved);
        if (!loaded) {
            warnings.push_back(owner + ".motion.keys_file: " + loaded.error());
        } else {
            spec.keys = std::move(*loaded);
        }
        if (!table.GetTableArray("keys").empty()) {
            warnings.push_back(owner +
                               ".motion: both `keys_file` and inline `keys` are given; the file "
                               "wins");
        }
        return;
    }

    for (const Config& keyTable : table.GetTableArray("keys")) {
        auto t = ReadDuration(keyTable, "t", owner + ".motion.keys", warnings);
        if (!t) {
            warnings.push_back(owner + ".motion.keys: an entry has no readable `t`; dropped");
            continue;
        }
        MotionKey key;
        key.t_s = *t;
        key.hasPosition = ReadVec3(keyTable, "position", key.position);

        glm::dvec3 euler(0.0);
        const auto quat = keyTable.GetDoubleArray("rotation_quat");
        if (quat.size() >= 4) {
            key.hasRotation = true;
            key.rotation = glm::normalize(glm::dquat(quat[3], quat[0], quat[1], quat[2]));
        } else if (ReadVec3(keyTable, "rotation_euler_degrees", euler)) {
            key.hasRotation = true;
            const glm::dvec3 r = glm::radians(euler);
            key.rotation = glm::angleAxis(r.x, glm::dvec3(1.0, 0.0, 0.0)) *
                           glm::angleAxis(r.y, glm::dvec3(0.0, 1.0, 0.0)) *
                           glm::angleAxis(r.z, glm::dvec3(0.0, 0.0, 1.0));
        }

        if (!key.hasPosition && !key.hasRotation) {
            warnings.push_back(owner + ".motion.keys: the key at t=" + std::to_string(key.t_s) +
                               " s carries neither a position nor a rotation; dropped");
            continue;
        }
        spec.keys.push_back(key);
    }
}

void ParseSegments(const Config& table, const String& owner, MotionSpec& spec,
                   Vector<String>& warnings) {
    for (const Config& segTable : table.GetTableArray("segments")) {
        auto from = ReadDuration(segTable, "from", owner + ".motion.segments", warnings);
        auto to = ReadDuration(segTable, "to", owner + ".motion.segments", warnings);
        if (!from || !to) {
            warnings.push_back(owner +
                               ".motion.segments: an entry needs both `from` and `to`; dropped");
            continue;
        }
        MotionSegment segment;
        segment.from_s = *from;
        segment.to_s = *to;
        segment.hasPosition =
            ReadExpr3(segTable, "position", segment.position, owner, warnings);
        segment.hasRotation =
            ReadExpr3(segTable, "rotation_euler_degrees", segment.rotationEulerDeg, owner,
                      warnings);
        if (!segment.hasPosition && !segment.hasRotation) {
            warnings.push_back(owner + ".motion.segments: the segment [" +
                               std::to_string(segment.from_s) + ", " +
                               std::to_string(segment.to_s) +
                               "] drives no channel; dropped");
            continue;
        }
        spec.segments.push_back(std::move(segment));
    }
}

bool ParseLocation(const Config& table, const String& owner, MotionLocation& out,
                   Vector<String>& warnings) {
    auto sub = table.GetTable("location");
    if (!sub) return false;
    const Config& location = *sub;

    const String type = location.GetString("type", "linear");
    if (type == "linear") {
        out.type = MotionLocationType::Linear;
        ReadVec3(location, "start", out.start);
        if (!ReadVec3(location, "velocity", out.velocity)) {
            // The heading form: degrees clockwise from +Z about +Y, which is
            // how a bearing is written on every map a scene author has seen.
            const f64 speed = location.GetDouble("speed_m_s", 0.0);
            const f64 heading = glm::radians(location.GetDouble("heading_deg", 0.0));
            out.velocity = glm::dvec3(speed * std::sin(heading), 0.0, speed * std::cos(heading));
        }
        return true;
    }
    if (type == "circle") {
        out.type = MotionLocationType::Circle;
        ReadVec3(location, "center", out.center);
        ReadVec3(location, "axis", out.axis);
        out.radius = location.GetDouble("radius", 0.0);
        out.startAngleDeg = location.GetDouble("start_angle_deg", 0.0);
        if (location.Has("angular_speed_deg_s")) {
            out.angularSpeedDeg_s = location.GetDouble("angular_speed_deg_s", 0.0);
        } else {
            const f64 period = location.GetDouble("period_s", 0.0);
            out.angularSpeedDeg_s = (period != 0.0) ? 360.0 / period : 0.0;
        }
        if (out.radius <= 0.0) {
            warnings.push_back(owner + ".motion.location: a circle needs a positive `radius`");
        }
        return true;
    }

    warnings.push_back(owner + ".motion.location.type: \"" + type +
                       "\" is not one of linear, circle; the location engine is ignored");
    return false;
}

bool ParseOrientation(const Config& table, const String& owner, MotionOrientation& out,
                      Vector<String>& warnings) {
    auto sub = table.GetTable("orientation");
    if (!sub) return false;
    const Config& orientation = *sub;

    ReadVec3(orientation, "forward", out.forward);
    ReadVec3(orientation, "up", out.up);

    const String type = orientation.GetString("type", "fixed");
    if (type == "fixed") {
        out.type = MotionOrientationType::Fixed;
        ReadVec3(orientation, "rotation_euler_degrees", out.eulerDeg);
        return true;
    }
    if (type == "spin") {
        out.type = MotionOrientationType::Spin;
        ReadVec3(orientation, "axis", out.axis);
        out.rateDeg_s = orientation.GetDouble("rate_deg_s", 0.0);
        return true;
    }
    if (type == "along_velocity") {
        out.type = MotionOrientationType::AlongVelocity;
        return true;
    }
    if (type == "look_at") {
        out.type = MotionOrientationType::LookAt;
        ReadVec3(orientation, "target", out.target);
        return true;
    }

    warnings.push_back(owner + ".motion.orientation.type: \"" + type +
                       "\" is not one of fixed, spin, along_velocity, look_at; the orientation "
                       "engine is ignored");
    return false;
}

}  // namespace

std::optional<MotionSpec> ParseMotionSpec(const Config& table, f64 origin_s, const String& owner,
                                          const MotionPathResolver& resolvePath,
                                          Vector<String>& warnings) {
    MotionSpec spec;
    spec.origin_s = origin_s;

    const String interpolation = table.GetString("interpolation", "linear");
    if (interpolation == "step") {
        spec.interpolation = MotionInterp::Step;
    } else if (interpolation == "linear") {
        spec.interpolation = MotionInterp::Linear;
    } else if (interpolation == "cubic") {
        spec.interpolation = MotionInterp::Cubic;
    } else {
        warnings.push_back(owner + ".motion.interpolation: \"" + interpolation +
                           "\" is not one of step, linear, cubic; using linear");
    }

    const String extrapolate = table.GetString("extrapolate", "hold");
    if (extrapolate == "hold") {
        spec.extrapolate = MotionExtrap::Hold;
    } else if (extrapolate == "loop") {
        spec.extrapolate = MotionExtrap::Loop;
    } else {
        warnings.push_back(owner + ".motion.extrapolate: \"" + extrapolate +
                           "\" is not one of hold, loop; using hold");
    }

    ParseKeys(table, owner, resolvePath, spec, warnings);
    ParseSegments(table, owner, spec, warnings);
    const bool hasLocation = ParseLocation(table, owner, spec.location, warnings);
    const bool hasOrientation = ParseOrientation(table, owner, spec.orientation, warnings);
    const bool hasEngine = hasLocation || hasOrientation;

    // Precedence, and a warning for anything the precedence discarded: a
    // config that says two things means one of them, and silence about which
    // is how a scene renders wrong for a week.
    const int forms = (spec.keys.empty() ? 0 : 1) + (spec.segments.empty() ? 0 : 1) +
                      (hasEngine ? 1 : 0);
    if (forms > 1) {
        warnings.push_back(owner +
                           ".motion: more than one of keys / segments / engines is present; "
                           "keys win over segments, which win over engines");
    }

    if (!spec.keys.empty()) {
        spec.form = MotionForm::Keys;
        spec.segments.clear();
        spec.location = MotionLocation{};
        spec.orientation = MotionOrientation{};
    } else if (!spec.segments.empty()) {
        spec.form = MotionForm::Segments;
        spec.location = MotionLocation{};
        spec.orientation = MotionOrientation{};
    } else if (hasEngine) {
        spec.form = MotionForm::Engine;
    } else {
        return std::nullopt;
    }

    return spec;
}

}  // namespace quantiloom::scene
