/**
 * @file ToolRegistry.hpp
 * @brief The catalogue of tools a host has offered
 *
 * A flat map, deliberately. The progressive-disclosure pattern -- a handful of
 * meta-tools standing in for a large catalogue, so `tools/list` stays small --
 * pays for itself somewhere north of thirty tools. Both hosts here are well
 * under that, and a flat list an agent can read in one go beats a search it has
 * to learn to drive.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"
#include "mcp/McpServer.hpp"

#include <mutex>
#include <unordered_map>

namespace quantiloom::mcp {

/**
 * @class ToolRegistry
 * @brief Name-to-tool map, readable from the transport thread
 */
class ToolRegistry {
public:
    /// @return An error if the name is empty or the schema is not a JSON object.
    Result<void, String> Add(const ToolDef& tool);

    /// @return nullptr when no tool has that name.
    [[nodiscard]] Optional<ToolDef> Find(StringView name) const;

    /// Tool descriptors as a JSON array, ordered by name so a client that
    /// caches the list sees a stable document.
    [[nodiscard]] String ListJson() const;

    [[nodiscard]] usize Size() const;

private:
    mutable std::mutex m_mutex;
    std::unordered_map<String, ToolDef> m_tools;
};

}  // namespace quantiloom::mcp
