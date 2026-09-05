#include "SequenceJob.hpp"

#include "RenderJob.hpp"
#include "core/Config.hpp"
#include "core/Log.hpp"
#include "renderer/OfflineRenderer.hpp"

#include <charconv>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>

namespace quantiloom::app {

namespace {

/// One `{name}` or `{name:format}` occurrence, already split.
struct Placeholder {
    String name;
    String format;
};

std::optional<Placeholder> SplitPlaceholder(StringView body) {
    const usize colon = body.find(':');
    Placeholder out;
    if (colon == StringView::npos) {
        out.name = String(body);
    } else {
        out.name = String(body.substr(0, colon));
        out.format = String(body.substr(colon + 1));
    }
    if (out.name.empty()) return std::nullopt;
    return out;
}

/// `05` -> five digits, zero padded. Anything else -> as many digits as it
/// takes, which is what a bare `{tick}` means.
String FormatTick(const String& format, i64 tick) {
    if (format.empty()) return std::to_string(tick);

    usize width = 0;
    const bool zeroPad = format[0] == '0';
    const auto [ptr, ec] =
        std::from_chars(format.data() + (zeroPad ? 1 : 0), format.data() + format.size(), width);
    if (ec != std::errc{} || width == 0 || width > 32) return std::to_string(tick);

    String digits = std::to_string(tick);
    const bool negative = !digits.empty() && digits[0] == '-';
    String body = negative ? digits.substr(1) : digits;
    while (body.size() < width) body.insert(body.begin(), zeroPad ? '0' : ' ');
    return negative ? ("-" + body) : body;
}

/// `.3f` -> three decimals. Bare -> the shortest form that round-trips, which
/// for a tick time is usually short.
String FormatTime(const String& format, f64 time_s) {
    char buffer[64];
    if (format.empty()) {
        std::snprintf(buffer, sizeof(buffer), "%g", time_s);
        return buffer;
    }
    // Only the .Nf family, because that is the only one a filename wants and
    // passing arbitrary text to printf would be a format-string hole.
    if (format.size() >= 2 && format.front() == '.' && format.back() == 'f') {
        usize decimals = 0;
        const auto [ptr, ec] = std::from_chars(format.data() + 1,
                                               format.data() + format.size() - 1, decimals);
        if (ec == std::errc{} && decimals <= 12) {
            std::snprintf(buffer, sizeof(buffer), "%.*f", static_cast<int>(decimals), time_s);
            return buffer;
        }
    }
    std::snprintf(buffer, sizeof(buffer), "%g", time_s);
    return buffer;
}

/// The default template: `renderer.output` with `_{tick:05}` before the suffix.
String DefaultTemplate(const String& outputPath) {
    const std::filesystem::path path(outputPath.empty() ? "frame.exr" : outputPath);
    const std::filesystem::path stem = path.parent_path() / path.stem();
    return stem.string() + "_{tick:05}" + path.extension().string();
}

}  // namespace

String FormatFrameName(const StringView tmpl, const i64 tick, const f64 time_s) {
    String out;
    out.reserve(tmpl.size() + 16);

    for (usize i = 0; i < tmpl.size(); ++i) {
        if (tmpl[i] != '{') {
            out.push_back(tmpl[i]);
            continue;
        }
        const usize close = tmpl.find('}', i);
        if (close == StringView::npos) {
            out.append(tmpl.substr(i));
            break;
        }
        const auto placeholder = SplitPlaceholder(tmpl.substr(i + 1, close - i - 1));
        if (!placeholder) {
            out.append(tmpl.substr(i, close - i + 1));
        } else if (placeholder->name == "tick") {
            out += FormatTick(placeholder->format, tick);
        } else if (placeholder->name == "time_s") {
            out += FormatTime(placeholder->format, time_s);
        } else {
            // Copied through rather than dropped: a name this does not know is
            // more likely a literal brace than a typo, and eating it would make
            // a file appear under a name nobody asked for.
            out.append(tmpl.substr(i, close - i + 1));
        }
        i = close;
    }
    return out;
}

int RunSequence(const SequenceOptions& options) {
    const std::filesystem::path configPath(options.configPath);

    auto configResult = Config::Load(configPath);
    if (!configResult.has_value()) {
        QL_LOG_ERROR("Failed to load configuration: {}", configResult.error());
        return 1;
    }
    const Config config = configResult.value();

    OfflineRenderer::InitParams init;
    init.atmosphereModelPackFallback = options.atmosphereModelPackFallback;
    init.baseDir = configPath.parent_path().string();

    // Built once. Everything expensive -- the device, the scene, the
    // acceleration structure, the thermal schedule and its steady state --
    // happens here and not per frame.
    auto rendererResult = OfflineRenderer::Create(config, init);
    if (!rendererResult.has_value()) {
        QL_LOG_ERROR("{}", rendererResult.error());
        return 1;
    }
    OfflineRenderer& renderer = *rendererResult.value();

    const TimelineInfo info = renderer.GetTimelineInfo();
    if (!info.present) {
        QL_LOG_ERROR("{} has no [timeline]; there is nothing to sequence. Render it as a "
                     "single frame instead.",
                     options.configPath);
        return 1;
    }

    const i64 lastTick = info.TickCount() - 1;
    const i64 from = std::max<i64>(options.fromTick.value_or(0), 0);
    const i64 to = std::min<i64>(options.toTick.value_or(lastTick), lastTick);
    const u32 every = std::max(options.every, 1u);
    if (to < from) {
        QL_LOG_ERROR("--from-tick {} is past --to-tick {}", from, to);
        return 1;
    }

    const String tmpl = options.outputTemplate.empty()
                            ? DefaultTemplate(renderer.Params().outputPath)
                            : options.outputTemplate;
    if (tmpl.find("{tick") == String::npos && tmpl.find("{time_s") == String::npos) {
        QL_LOG_WARN("The output template names neither {{tick}} nor {{time_s}}, so every "
                    "frame will overwrite the last one: {}", tmpl);
    }

    i64 frameCount = 0;
    for (i64 tick = from; tick <= to; tick += every) ++frameCount;

    QL_LOG_INFO("Sequence: {} frame(s), tick {} to {} every {}, {} tick/s over {:.3f} s",
                frameCount, from, to, every, info.ticksPerSecond, info.end_s - info.start_s);

    if (options.dryRun) {
        for (i64 tick = from; tick <= to; tick += every) {
            const f64 t = info.TimeOfTick(tick);
            std::cout << "tick " << tick << "  t=" << t << " s  -> "
                      << FormatFrameName(tmpl, tick, t) << "\n";
        }
        return 0;
    }

    i64 index = 0;
    u32 failures = 0;
    const auto sequenceStarted = std::chrono::steady_clock::now();

    for (i64 tick = from; tick <= to; tick += every) {
        ++index;
        const f64 t = info.TimeOfTick(tick);
        const auto frameStarted = std::chrono::steady_clock::now();

        if (auto moved = renderer.SetTimelineTime(t); !moved.has_value()) {
            QL_LOG_ERROR("Frame {}/{} (tick {}): {}", index, frameCount, tick, moved.error());
            ++failures;
            continue;
        }

        RenderOutcome outcome;
        outcome.width = renderer.Params().width;
        outcome.height = renderer.Params().height;
        outcome.spp = renderer.Params().spp;
        outcome.modeName = renderer.Params().modeName;
        outcome.wavelengthNm = renderer.Params().wavelengthNm;
        outcome.exrPath = FormatFrameName(tmpl, tick, t);

        // Named before it is written, so a template with a directory in it does
        // not fail on the first frame for want of one.
        const std::filesystem::path parent = std::filesystem::path(outcome.exrPath).parent_path();
        if (!parent.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(parent, ec);
        }

        OfflineRenderOutput rendered = renderer.Render();
        if (!rendered.error.empty()) {
            QL_LOG_ERROR("Frame {}/{} (tick {}): {}", index, frameCount, tick, rendered.error);
            ++failures;
            continue;
        }
        if (!rendered.wroteItsOwnOutput) {
            WriteFrameOutputs(config, rendered, renderer.Params().mode, outcome);
        }
        if (!outcome.error.empty()) ++failures;

        const TimelineInfo now = renderer.GetTimelineInfo();
        const f64 seconds =
            std::chrono::duration<f64>(std::chrono::steady_clock::now() - frameStarted).count();
        if (now.thermalMapped) {
            QL_LOG_INFO("Frame {}/{} (tick {}, t={:.3f} s, hour {:.3f}, epoch {}/{}) -> {}  "
                        "{:.1f} s",
                        index, frameCount, tick, t, now.currentThermalHour,
                        now.currentThermalEpoch + 1, now.thermalEpochCount, outcome.exrPath,
                        seconds);
        } else {
            QL_LOG_INFO("Frame {}/{} (tick {}, t={:.3f} s) -> {}  {:.1f} s", index, frameCount,
                        tick, t, outcome.exrPath, seconds);
        }
    }

    const f64 total =
        std::chrono::duration<f64>(std::chrono::steady_clock::now() - sequenceStarted).count();
    if (failures == 0) {
        QL_LOG_INFO("Sequence complete: {} frame(s) in {:.1f} s", frameCount, total);
        return 0;
    }
    QL_LOG_ERROR("Sequence finished with {} of {} frame(s) failed, in {:.1f} s", failures,
                 frameCount, total);
    return 1;
}

}  // namespace quantiloom::app
