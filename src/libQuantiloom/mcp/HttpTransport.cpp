#include "mcp/HttpTransport.hpp"

#include "core/Log.hpp"
#include "mcp/Protocol.hpp"

QL_DISABLE_WARNINGS_PUSH
#include <httplib.h>
QL_DISABLE_WARNINGS_POP

#include <algorithm>
#include <cctype>

namespace quantiloom::mcp {
namespace {

constexpr const char* kBindAddress = "127.0.0.1";
constexpr const char* kEndpoint = "/mcp";

/// Strips the scheme and port from an Origin or Host header and asks whether
/// what is left names this machine.
bool IsLoopbackAuthority(String value) {
    if (value.empty()) {
        return true;  // absent header: a non-browser client, nothing to spoof
    }

    // http://127.0.0.1:8765 -> 127.0.0.1:8765
    const usize scheme = value.find("://");
    if (scheme != String::npos) {
        value = value.substr(scheme + 3);
    }
    // Trim a path if one came along.
    const usize slash = value.find('/');
    if (slash != String::npos) {
        value = value.substr(0, slash);
    }

    // [::1]:8765 -> ::1
    if (!value.empty() && value.front() == '[') {
        const usize close = value.find(']');
        if (close == String::npos) {
            return false;
        }
        value = value.substr(1, close - 1);
    } else {
        const usize colon = value.rfind(':');
        if (colon != String::npos) {
            value = value.substr(0, colon);
        }
    }

    std::transform(value.begin(), value.end(), value.begin(),
                   [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });

    return value == "127.0.0.1" || value == "localhost" || value == "::1";
}

/**
 * @brief Refuse anything that did not come from this machine
 *
 * A page in a browser can POST to http://127.0.0.1 from any origin it likes;
 * without this check, visiting a hostile site while Studio is running would let
 * that site drive the renderer. Binding loopback stops the network, and this
 * stops the browser.
 */
bool IsRequestLocal(const httplib::Request& req) {
    return IsLoopbackAuthority(req.get_header_value("Origin")) &&
           IsLoopbackAuthority(req.get_header_value("Host"));
}

}  // namespace

HttpTransport::HttpTransport() = default;

HttpTransport::~HttpTransport() {
    Stop();
}

Result<std::unique_ptr<HttpTransport>, String> HttpTransport::Create(const u16 port,
                                                                    Protocol& protocol) {
    auto transport = std::unique_ptr<HttpTransport>(new HttpTransport());
    transport->m_server = std::make_unique<httplib::Server>();
    httplib::Server& server = *transport->m_server;

    server.Post(kEndpoint, [&protocol](const httplib::Request& req, httplib::Response& res) {
        if (!IsRequestLocal(req)) {
            QL_LOG_WARN("MCP: rejected a request with non-loopback Origin/Host ('{}' / '{}')",
                        req.get_header_value("Origin"), req.get_header_value("Host"));
            res.status = 403;
            res.set_content(R"({"error":"only loopback origins are accepted"})",
                            "application/json");
            return;
        }

        const Protocol::Response response = protocol.HandlePost(req.body);
        res.status = static_cast<int>(response.httpStatus);
        // A client that announced its revision gets it echoed, which is what the
        // header is for; one that said nothing is talking to the newest we know.
        const String version = req.get_header_value("MCP-Protocol-Version");
        res.set_header("MCP-Protocol-Version", version.empty() ? kLatestProtocolVersion : version);
        if (response.body.empty()) {
            res.set_content("", "application/json");
        } else {
            res.set_content(response.body, "application/json");
        }
    });

    // Streamable HTTP's GET carries server-initiated messages. There are none.
    server.Get(kEndpoint, [](const httplib::Request&, httplib::Response& res) {
        res.status = 405;
        res.set_header("Allow", "POST");
        res.set_content(R"({"error":"this server does not send server-initiated messages"})",
                        "application/json");
    });

    if (!server.bind_to_port(kBindAddress, port)) {
        return Result<std::unique_ptr<HttpTransport>, String>::Err(
            "could not bind " + String(kBindAddress) + ":" + std::to_string(port) +
            " -- another server may already be using that port");
    }
    transport->m_port = port;

    transport->m_thread = std::thread([&server] {
        // Returns when the server is stopped; a false return here means the
        // socket died, which Stop() also causes.
        server.listen_after_bind();
    });

    QL_LOG_INFO("MCP: listening on http://{}:{}{}", kBindAddress, port, kEndpoint);
    return transport;
}

void HttpTransport::Stop() {
    if (m_server) {
        m_server->stop();
    }
    if (m_thread.joinable()) {
        m_thread.join();
    }
    m_server.reset();
}

}  // namespace quantiloom::mcp
