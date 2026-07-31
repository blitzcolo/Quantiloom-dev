#include "mcp/Protocol.hpp"

#include "core/Log.hpp"
#include "mcp/CommandQueue.hpp"
#include "mcp/ToolRegistry.hpp"

QL_DISABLE_WARNINGS_PUSH
#include <nlohmann/json.hpp>
QL_DISABLE_WARNINGS_POP

#include <array>
#include <memory>
#include <utility>

namespace quantiloom::mcp {
namespace {

using json = nlohmann::json;

// JSON-RPC 2.0 error codes. -32000..-32019 are reserved for implementations;
// nothing here needs one yet.
constexpr i32 kParseError = -32700;
constexpr i32 kInvalidRequest = -32600;
constexpr i32 kMethodNotFound = -32601;
constexpr i32 kInvalidParams = -32602;

/// Revisions this server will echo back when a client asks for one of them.
/// Ordered newest first, which is also the order server/discover reports.
constexpr std::array<const char*, 4> kSupportedVersions = {
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
};

bool IsSupportedVersion(const String& version) {
    for (const char* known : kSupportedVersions) {
        if (version == known) {
            return true;
        }
    }
    return false;
}

json MakeError(const json& id, const i32 code, const String& message) {
    return json{
        {"jsonrpc", "2.0"},
        {"id", id},
        {"error", {{"code", code}, {"message", message}}},
    };
}

json MakeResult(const json& id, json result) {
    return json{
        {"jsonrpc", "2.0"},
        {"id", id},
        {"result", std::move(result)},
    };
}

json ServerCapabilities() {
    // No resources, no prompts, no logging. Advertising a capability this
    // server does not implement is how a client ends up calling a method that
    // answers -32601 in the middle of a task.
    return json{{"tools", {{"listChanged", false}}}};
}

/// The tool result content array, in the shape 2025-06-18 fixed and later
/// revisions kept.
json ToolResultJson(const ToolResult& result) {
    json content = json::array();
    if (!result.text.empty()) {
        content.push_back({{"type", "text"}, {"text", result.text}});
    }
    if (!result.imageBase64.empty()) {
        // Native image content, not text. A downsampled 768px PNG costs an agent
        // roughly a thousand tokens as an image block and tens of thousands as
        // a base64 string, and the difference is entirely in whether the type
        // says "image".
        content.push_back({
            {"type", "image"},
            {"data", result.imageBase64},
            {"mimeType", result.imageMimeType.empty() ? String("image/png") : result.imageMimeType},
        });
    }
    if (content.empty()) {
        content.push_back({{"type", "text"}, {"text", ""}});
    }

    return json{{"content", std::move(content)}, {"isError", result.isError}};
}

}  // namespace

Protocol::Protocol(ToolRegistry& registry, CommandQueue& queue, String serverName,
                   String serverVersion)
    : m_registry(registry),
      m_queue(queue),
      m_serverName(std::move(serverName)),
      m_serverVersion(std::move(serverVersion)) {}

Protocol::Response Protocol::HandlePost(const String& body) {
    json request;
    try {
        request = json::parse(body);
    } catch (const json::exception& e) {
        QL_LOG_WARN("MCP: unparseable request body: {}", e.what());
        return {400, MakeError(nullptr, kParseError, String("parse error: ") + e.what()).dump()};
    }

    // JSON-RPC batching was removed in 2025-06-18 and never reinstated.
    if (!request.is_object()) {
        return {400, MakeError(nullptr, kInvalidRequest,
                               "expected a single JSON-RPC object; batching is not supported")
                         .dump()};
    }

    const String method = request.value("method", String());
    if (method.empty()) {
        return {400, MakeError(request.value("id", json(nullptr)), kInvalidRequest,
                               "missing 'method'")
                         .dump()};
    }

    // No "id" means a notification: acknowledged with 202 and no body.
    const bool isNotification = !request.contains("id") || request["id"].is_null();
    const json id = isNotification ? json(nullptr) : request["id"];
    const json params = request.value("params", json::object());

    if (isNotification) {
        // notifications/initialized, notifications/cancelled and friends. There
        // is nothing to keep: the handshake is not a gate here.
        return {202, String()};
    }

    if (method == "initialize") {
        const String requested = params.value("protocolVersion", String());
        const String negotiated =
            IsSupportedVersion(requested) ? requested : String(kLatestProtocolVersion);
        if (!requested.empty() && negotiated != requested) {
            QL_LOG_INFO("MCP: client asked for protocol {}, answering {}", requested, negotiated);
        }
        return {200, MakeResult(id, json{
                                        {"protocolVersion", negotiated},
                                        {"capabilities", ServerCapabilities()},
                                        {"serverInfo",
                                         {{"name", m_serverName}, {"version", m_serverVersion}}},
                                    })
                         .dump()};
    }

    if (method == "server/discover") {
        // 2026-07-28's replacement for the handshake. Answered so a newer client
        // can see what it is talking to, not as a claim of conformance.
        json versions = json::array();
        for (const char* v : kSupportedVersions) {
            versions.push_back(v);
        }
        return {200, MakeResult(id, json{
                                        {"protocolVersions", std::move(versions)},
                                        {"capabilities", ServerCapabilities()},
                                        {"serverInfo",
                                         {{"name", m_serverName}, {"version", m_serverVersion}}},
                                    })
                         .dump()};
    }

    if (method == "ping") {
        return {200, MakeResult(id, json::object()).dump()};
    }

    if (method == "tools/list") {
        json result;
        result["tools"] = json::parse(m_registry.ListJson());
        // The catalogue only changes when the host registers something, which
        // does not happen while a client is connected. Let it cache for a
        // minute rather than re-fetching around every call.
        result["ttlMs"] = 60000;
        return {200, MakeResult(id, std::move(result)).dump()};
    }

    if (method == "tools/call") {
        const String name = params.value("name", String());
        if (name.empty()) {
            return {200, MakeError(id, kInvalidParams, "missing tool name").dump()};
        }
        const Optional<ToolDef> tool = m_registry.Find(name);
        if (!tool.has_value()) {
            return {200, MakeError(id, kInvalidParams, "no such tool: " + name).dump()};
        }

        const String argumentsJson = params.contains("arguments") ? params["arguments"].dump()
                                                                  : String("{}");

        // The handler runs on the host's thread, not this one. The result has to
        // outlive a timeout -- the work may still run after we stop waiting --
        // so it lives in a block both sides hold a reference to.
        auto slot = std::make_shared<ToolResult>();
        const ToolDef& def = tool.value();
        const bool ran = m_queue.Submit(
            [handler = def.handler, argumentsJson, slot]() { *slot = handler(argumentsJson); },
            def.venue == ToolVenue::HostDispatch, def.timeoutMs);

        if (!ran) {
            QL_LOG_WARN("MCP: tool '{}' timed out after {} ms", name, def.timeoutMs);
            return {200, MakeResult(id, ToolResultJson(ToolResult::Error(
                                            "The host did not run this call within " +
                                            std::to_string(def.timeoutMs) +
                                            " ms. It may still be busy -- check that the "
                                            "application is responding, then retry.")))
                             .dump()};
        }

        return {200, MakeResult(id, ToolResultJson(*slot)).dump()};
    }

    return {200, MakeError(id, kMethodNotFound, "unsupported method: " + method).dump()};
}

}  // namespace quantiloom::mcp
