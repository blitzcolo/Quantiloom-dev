#include "mcp/McpServer.hpp"

#include "core/Log.hpp"
#include "mcp/CommandQueue.hpp"
#include "mcp/HttpTransport.hpp"
#include "mcp/Protocol.hpp"
#include "mcp/ToolRegistry.hpp"

#include <utility>

namespace quantiloom::mcp {

struct Server::Impl {
    ToolRegistry registry;
    CommandQueue queue;
    // Declaration order matters on the way out: the transport's thread calls
    // into the protocol, so the transport is destroyed first.
    std::unique_ptr<Protocol> protocol;
    std::unique_ptr<HttpTransport> transport;
    bool stopped = false;
};

Server::Server() : m_impl(std::make_unique<Impl>()) {}

Server::~Server() {
    Stop();
}

Result<std::unique_ptr<Server>, String> Server::Create(const ServerOptions& options) {
    auto server = std::unique_ptr<Server>(new Server());
    Impl& impl = *server->m_impl;

    impl.queue.SetWakeCallback(options.onCommandQueued);
    impl.queue.SetHostDispatch(options.hostDispatch);

    impl.protocol = std::make_unique<Protocol>(impl.registry, impl.queue, options.serverName,
                                               options.serverVersion);

    auto transport = HttpTransport::Create(options.port, *impl.protocol);
    if (!transport) {
        return Result<std::unique_ptr<Server>, String>::Err(transport.error());
    }
    impl.transport = std::move(transport.value());

    return server;
}

Result<void, String> Server::RegisterTool(const ToolDef& tool) {
    return m_impl->registry.Add(tool);
}

void Server::Pump() {
    m_impl->queue.Drain();
}

u32 Server::PendingCommands() const {
    return m_impl->queue.Pending();
}

u16 Server::Port() const {
    return m_impl->transport ? m_impl->transport->Port() : 0;
}

void Server::Stop() {
    if (m_impl->stopped) {
        return;
    }
    m_impl->stopped = true;

    // Transport first: no new work arrives after this returns. Then release
    // whoever is still waiting -- their handlers are not run, because the host
    // that owns what they would touch is on its way out.
    if (m_impl->transport) {
        m_impl->transport->Stop();
        m_impl->transport.reset();
    }
    m_impl->queue.Shutdown();
    m_impl->protocol.reset();

    QL_LOG_INFO("MCP: server stopped");
}

}  // namespace quantiloom::mcp
