/**
 * @file HttpTransport.hpp
 * @brief Streamable HTTP transport, bound to loopback
 *
 * One endpoint, POST only. The GET half of Streamable HTTP exists to carry
 * server-initiated messages over SSE; this server initiates nothing, so GET
 * answers 405 rather than holding a stream open that would never carry
 * anything. Clients handle that -- it is the documented shape for a server
 * without server-initiated messages.
 *
 * The listener runs on its own thread and calls into Protocol, which is why
 * everything Protocol touches is either immutable or behind a mutex.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"

#include <memory>
#include <thread>

namespace httplib {
class Server;
}

namespace quantiloom::mcp {

class Protocol;

/**
 * @class HttpTransport
 * @brief Owns the socket and the thread that reads it
 */
class HttpTransport {
public:
    /**
     * @brief Bind the port, then start serving
     *
     * The bind happens here, before the thread starts, so "port already in use"
     * is an error the host can show at start-up instead of a server that seems
     * to be running and never answers.
     */
    static Result<std::unique_ptr<HttpTransport>, String> Create(u16 port, Protocol& protocol);

    ~HttpTransport();

    HttpTransport(const HttpTransport&) = delete;
    HttpTransport& operator=(const HttpTransport&) = delete;

    void Stop();

    [[nodiscard]] u16 Port() const { return m_port; }

private:
    HttpTransport();

    std::unique_ptr<httplib::Server> m_server;
    std::thread m_thread;
    u16 m_port = 0;
};

}  // namespace quantiloom::mcp
