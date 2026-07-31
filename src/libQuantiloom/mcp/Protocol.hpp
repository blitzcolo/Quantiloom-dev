/**
 * @file Protocol.hpp
 * @brief JSON-RPC 2.0 dispatch for the MCP methods this server answers
 *
 * Hand-written, because there is no official C++ SDK and the community ones all
 * sit at least two revisions behind the specification. What is actually needed
 * here is small: a handshake, a catalogue, a call, and a ping.
 *
 * **Which revision.** The clients in the field -- Claude Code, MCP Inspector --
 * still speak the 2025 line, where a session opens with `initialize` and the
 * negotiated version comes back in the result. That is the path implemented in
 * full. The 2026-07-28 revision drops the handshake and makes every request
 * self-describing; rather than implement a second protocol for a client that
 * does not exist yet, this simply does not *require* the handshake. A request
 * that arrives cold is served. `server/discover` is answered so a newer client
 * can find out what it is talking to. Neither costs anything, and neither
 * claims conformance the code has not earned.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"

namespace quantiloom::mcp {

class ToolRegistry;
class CommandQueue;

/// Reported when a client asks for a revision this server does not know.
inline constexpr const char* kLatestProtocolVersion = "2025-11-25";

/**
 * @class Protocol
 * @brief Turns one HTTP request body into one response body
 */
class Protocol {
public:
    struct Response {
        i32 httpStatus = 200;
        /// Empty for a notification, which carries no reply.
        String body;
    };

    Protocol(ToolRegistry& registry, CommandQueue& queue, String serverName, String serverVersion);

    /// @param body The POST body, expected to be a single JSON-RPC message.
    Response HandlePost(const String& body);

private:
    ToolRegistry& m_registry;
    CommandQueue& m_queue;
    String m_serverName;
    String m_serverVersion;
};

}  // namespace quantiloom::mcp
