#pragma once

#include "core/Config.hpp"
#include "core/Types.hpp"

#include <filesystem>
#include <vector>

namespace quantiloom::app {

/**
 * @brief One line of a batch manifest: a configuration, and what to change in it
 *
 * A manifest was a list of paths, one per line, and that is still what most of
 * them are. A line may now also carry overrides:
 *
 *   scenes/plate.toml
 *   scenes/plate.toml | material_overrides.Plate.ir_temperature_k=300.0
 *   scenes/plate.toml | @noon.toml renderer.output="seq/noon.exr"
 *
 * Everything left of the `|` is the path, exactly as before -- a line without
 * one is read the way it always was. To the right, whitespace-separated:
 *
 *   key.path=value   a TOML scalar, dotted key, layered onto the config
 *   @file.toml       a sidecar document, layered whole
 *
 * `|` is the separator because Windows forbids it in a filename, so no path
 * can be mistaken for one. Values follow TOML's own grammar: bare numbers and
 * booleans, quoted strings. Quotes protect spaces, which is the only reason a
 * token would contain one.
 *
 * Layering order is config, then --override, then this line's sidecars in the
 * order written, then its inline pairs. Later wins. Note that
 * Config::MergedWith replaces arrays whole, so a per-line override of one
 * material must use [material_overrides.<name>] rather than [[materials]].
 */
struct ManifestEntry {
    /// Absolute, resolved against the manifest's own directory.
    std::filesystem::path configPath;
    /// Absolute paths of `@sidecar.toml` tokens, in the order written.
    std::vector<std::filesystem::path> sidecars;
    /// The inline `key=value` pairs, already assembled into a TOML document.
    /// Empty when the line had none.
    String inlineOverrides;
    /// 1-based line number in the manifest, for an error that has to say where.
    usize lineNumber = 0;
};

/**
 * @brief Parse a manifest into entries
 *
 * Blank lines and `#` comments are skipped. Relative paths -- the config's and
 * any sidecar's -- resolve against the manifest's own directory, so a list can
 * sit beside the configs it names and travel with them.
 *
 * A malformed override is an error for the whole manifest rather than for one
 * job: a typo that silently rendered the unmodified scene would be a sequence
 * with a wrong frame in it, and nothing about the output would say so.
 *
 * @return the entries, or why the manifest could not be read
 */
[[nodiscard]] Result<std::vector<ManifestEntry>, String> ParseManifest(
    const std::filesystem::path& manifestPath);

/**
 * @brief Layer an entry's own overrides onto a configuration
 *
 * Sidecars first, in the order written, then the inline pairs -- so the most
 * specific thing on the line wins.
 *
 * @return the merged configuration, or why a sidecar could not be read
 */
[[nodiscard]] Result<Config, String> ApplyEntryOverrides(const Config& config,
                                                         const ManifestEntry& entry);

/// A one-line summary of what an entry overrides, for the dry run.
[[nodiscard]] String DescribeEntryOverrides(const ManifestEntry& entry);

}  // namespace quantiloom::app
