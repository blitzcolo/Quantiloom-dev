/**
 * @file McpServer.hpp
 * @brief Model Context Protocol server: an agent-facing front end for a host
 *
 * Speaks MCP over Streamable HTTP on the loopback interface, so an agent
 * (Claude Code, MCP Inspector, any client that reads a `.mcp.json`) can drive
 * whichever Quantiloom host is running. The protocol, the JSON-RPC framing and
 * the transport live inside the library; what a given host *offers* does not.
 *
 * **The host registers the tools.** Nothing is exposed by default. This is not
 * timidity about the API surface -- the two hosts genuinely differ. The CLI is
 * stateless: a configuration arrives, a frame comes back, nothing persists
 * between calls, which is exactly what OfflineRenderer already is. Studio is a
 * document editor: a scene stays loaded, an agent nudges a light and asks what
 * changed. A single built-in tool set would have to pretend one of those is the
 * other. Registration makes the asymmetry the plain fact it is: the difference
 * between the hosts is the difference between what each one called RegisterTool
 * with.
 *
 * **Handlers do not run on the transport thread.** They run on whichever thread
 * calls Pump(), one at a time, in submission order. The renderer synchronises
 * nothing of its own -- ExternalRenderContext's setters write their fields and
 * touch GPU buffers on the calling thread -- so a tool handler that ran on the
 * HTTP thread would be racing the frame in flight. Pump() is what makes a
 * handler's access to the host as safe as the host's own.
 *
 * Usage:
 * @code
 * mcp::ServerOptions options;
 * options.port = 8600;
 * options.serverVersion = QUANTILOOM_VERSION_STRING;
 * options.onCommandQueued = [this] { requestUpdate(); };  // wake the pump
 *
 * auto server = mcp::Server::Create(options);
 * if (!server) { QL_LOG_ERROR("{}", server.error()); return; }
 * m_server = std::move(server.value());
 *
 * mcp::ToolDef status;
 * status.name = "ql_get_status";
 * status.description = "Current render state: accumulated samples, ...";
 * status.inputSchemaJson = R"({"type":"object","properties":{}})";
 * status.readOnly = true;
 * status.handler = [this](const String&) { return mcp::ToolResult::Text(...); };
 * m_server->RegisterTool(status);
 *
 * // ... per frame, on the render thread:
 * m_server->Pump();
 * @endcode
 *
 * @note Binds 127.0.0.1 and rejects requests whose Origin or Host is not
 *       loopback. There is no authentication, and no switch to turn the
 *       loopback restriction off: anything reachable from another machine
 *       would need one, and this does not have one.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Image.hpp"
#include "core/Platform.hpp"
#include "core/Types.hpp"

#include <functional>
#include <memory>

namespace quantiloom::mcp {

/**
 * @brief Where a tool's handler is allowed to run
 *
 * Pump() is called from inside the host's frame callback, which is the safe
 * point for anything that touches the renderer. It is *not* a safe point for
 * anything that re-enters the host's event loop -- Studio's scene load pumps
 * Qt events while shaders compile, and doing that from within a frame callback
 * is the re-entrancy that has already cost this codebase a crash. A tool that
 * does such work says so, and Pump() hands it to the host to schedule instead
 * of running it.
 */
enum class ToolVenue : u32 {
    /// Run inline in Pump(). The default, and correct for anything that only
    /// reads or writes renderer state.
    FrameBoundary = 0,

    /// Hand to ServerOptions::hostDispatch, to be run once the host is back in
    /// its own event loop. For document-level work: loading a scene, applying a
    /// whole configuration.
    HostDispatch = 1,
};

/**
 * @brief What a tool handler returns
 *
 * Text is what an agent should read most of the time -- it is cheap, and a
 * number it can compare beats an image it has to look at. The image field is
 * for when looking is the point; it becomes MCP image content, which the host
 * turns into a native image block rather than a wall of base64.
 */
struct ToolResult {
    /// Human-readable text, or a JSON document as text. Both are common in the
    /// wild; JSON reads better when the agent needs to compare fields.
    String text;

    /// Base64-encoded image bytes. Empty for the usual text-only result.
    String imageBase64;

    /// MIME type of `imageBase64`, e.g. "image/png". Ignored when there is no
    /// image.
    String imageMimeType;

    /// Marks the call as failed. The text is still delivered -- an agent can
    /// act on "no scene is loaded; call ql_apply_config first" and cannot act
    /// on a transport-level error, so failures are reported here rather than
    /// as JSON-RPC errors.
    bool isError = false;

    static ToolResult Text(String message) {
        ToolResult r;
        r.text = std::move(message);
        return r;
    }

    static ToolResult Error(String message) {
        ToolResult r;
        r.text = std::move(message);
        r.isError = true;
        return r;
    }
};

/// Receives the tool call's `arguments` object as a JSON document.
using ToolHandler = std::function<ToolResult(const String& argumentsJson)>;

/**
 * @brief Encode a frame for ToolResult::imageBase64
 *
 * Downsamples so the long edge is at most `maxDimension`, sRGB-encodes it the
 * way ImageIO::WritePNG does, and base64s the PNG. Both hosts go through this
 * rather than each reaching for its own image library, so a frame looks the
 * same to an agent whichever one produced it.
 *
 * The default size is a cost decision. An image block costs a model roughly
 * width x height / 750 tokens, so a 1080p frame is about 2,800 and a 768px one
 * under a thousand -- and nothing anybody asks a renderer about is invisible at
 * 768. Values are clamped to [0,1]: a linear HDR frame with no exposure applied
 * will clip, which is itself worth seeing.
 *
 * @return Base64 PNG, or an empty string if the image was empty or unencodable.
 */
[[nodiscard]] QL_API String EncodeImageContent(const Image& image, u32 maxDimension = 768);

/**
 * @brief One tool, as the agent sees it and as the host implements it
 */
struct ToolDef {
    /// Protocol identifier. Stable, lowercase, prefixed -- `ql_set_lighting`.
    /// Never translated: this is an API name, not a label.
    String name;

    /// Written for the agent, as API documentation: what it does, when to use
    /// it, when not to, what comes back. This is prompt engineering with a
    /// different file extension.
    String description;

    /// JSON Schema for the arguments object, as text. Parsed once at
    /// registration; a document that does not parse is rejected.
    String inputSchemaJson;

    ToolHandler handler;

    ToolVenue venue = ToolVenue::FrameBoundary;

    /// Advertised as `readOnlyHint`. A tool that only reads host state.
    bool readOnly = false;

    /// Advertised as `destructiveHint`. A client may ask the user first.
    bool destructive = false;

    /// How long the transport thread waits for Pump() to run this before
    /// giving up and answering with a timeout. Generous defaults are wrong
    /// here in both directions: too short and a legitimate scene load reports
    /// failure while still running, too long and a stalled host holds a
    /// connection open for minutes.
    u32 timeoutMs = 30000;
};

/**
 * @brief Host-side settings, fixed at Create()
 */
struct ServerOptions {
    /// Loopback port.
    ///
    /// 8600 rather than something in the 876x range the hosts used to pick:
    /// Windows reserves blocks of the ephemeral range for Hyper-V and WinNAT,
    /// and on a machine whose reservations covered 8725-8824 the bind failed
    /// with a message blaming another server. `netsh int ipv4 show
    /// excludedportrange protocol=tcp` lists them.
    ///
    /// Every host now defaults here, so two of them cannot serve at once; the
    /// second reports the bind failure and the host's own switch -- `serve
    /// --port`, Studio's `--mcp=PORT` -- moves it.
    u16 port = 8600;

    /// Reported in `serverInfo`. Distinguishes hosts to the agent.
    String serverName = "quantiloom";

    String serverVersion;

    /**
     * @brief Called from the transport thread when work arrives
     *
     * A host that only pumps when it happens to be drawing will not answer a
     * tool call while it is idle -- Studio stops requesting frames when the
     * viewport is paused, and would simply never get to the queue. This is the
     * nudge that says "there is something to run". It is called on the
     * transport thread, so it must be safe there: post an event, set a flag,
     * notify a condition variable. Do not touch the renderer in it.
     */
    std::function<void()> onCommandQueued;

    /**
     * @brief Runs a HostDispatch tool outside the frame callback
     *
     * Given a callable, the host arranges for it to run later on the host's own
     * thread -- Qt posts it with a queued connection. When this is empty,
     * HostDispatch tools run inline in Pump() like every other tool, which is
     * what a headless host with no event loop to re-enter wants.
     */
    std::function<void(std::function<void()>)> hostDispatch;
};

/**
 * @class Server
 * @brief An MCP server for one host process
 */
class QL_API Server {
public:
    /**
     * @brief Bind the port and start listening
     *
     * Binding happens here rather than on the listener thread so that a port
     * already in use is an error the host can report, instead of a server that
     * silently never answers.
     *
     * @return The server, or why it could not start.
     */
    static Result<std::unique_ptr<Server>, String> Create(const ServerOptions& options);

    /// Stops the transport and fails anything still queued.
    ~Server();

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    /**
     * @brief Add a tool to the catalogue
     *
     * Registering a name twice replaces the earlier definition. Call before or
     * after clients connect -- `tools/list` is answered from the current
     * catalogue each time.
     *
     * @return An error if the name is empty or the schema does not parse.
     */
    Result<void, String> RegisterTool(const ToolDef& tool);

    /**
     * @brief Run the tool calls that have arrived
     *
     * Call from the host's frame callback, or from a headless host's main loop.
     * Runs every queued FrameBoundary command to completion, in order, then
     * returns. Handlers run on the calling thread.
     */
    void Pump();

    /// How many calls are waiting. Zero most of the time.
    [[nodiscard]] u32 PendingCommands() const;

    /// The port actually bound.
    [[nodiscard]] u16 Port() const;

    /// Stop serving. Idempotent; the destructor calls it.
    void Stop();

private:
    Server();

    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace quantiloom::mcp
