#include "mcp/ToolRegistry.hpp"

#include <algorithm>

QL_DISABLE_WARNINGS_PUSH
#include <nlohmann/json.hpp>
QL_DISABLE_WARNINGS_POP

namespace quantiloom::mcp {

using json = nlohmann::json;

Result<void, String> ToolRegistry::Add(const ToolDef& tool) {
    if (tool.name.empty()) {
        return Result<void, String>::Err("tool name is empty");
    }
    if (!tool.handler) {
        return Result<void, String>::Err("tool '" + tool.name + "' has no handler");
    }

    // Parse the schema now rather than on every tools/list. A malformed schema
    // that only surfaced when an agent asked for the catalogue would take the
    // whole catalogue down with it.
    json schema;
    try {
        schema = json::parse(tool.inputSchemaJson);
    } catch (const json::exception& e) {
        return Result<void, String>::Err("tool '" + tool.name + "' has an unparseable input schema: " +
                                         e.what());
    }
    if (!schema.is_object()) {
        return Result<void, String>::Err("tool '" + tool.name + "' input schema is not a JSON object");
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_tools[tool.name] = tool;
    return {};
}

Optional<ToolDef> ToolRegistry::Find(const StringView name) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto it = m_tools.find(String(name));
    if (it == m_tools.end()) {
        return std::nullopt;
    }
    return it->second;
}

String ToolRegistry::ListJson() const {
    Vector<const ToolDef*> ordered;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        ordered.reserve(m_tools.size());
        for (const auto& [name, tool] : m_tools) {
            ordered.push_back(&tool);
        }
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const ToolDef* a, const ToolDef* b) { return a->name < b->name; });

    json tools = json::array();
    for (const ToolDef* tool : ordered) {
        json entry;
        entry["name"] = tool->name;
        entry["description"] = tool->description;
        entry["inputSchema"] = json::parse(tool->inputSchemaJson);
        entry["annotations"] = {
            {"readOnlyHint", tool->readOnly},
            {"destructiveHint", tool->destructive},
        };
        tools.push_back(std::move(entry));
    }
    return tools.dump();
}

usize ToolRegistry::Size() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_tools.size();
}

}  // namespace quantiloom::mcp
