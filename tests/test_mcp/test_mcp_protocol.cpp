// ============================================================================
// Quantiloom - Unit Tests for mcp/Protocol.hpp
// ============================================================================
// Tests cover:
// - JSON-RPC framing: parse errors, missing method, notifications, batching
// - initialize: version negotiation, capabilities, serverInfo
// - tools/list: catalogue shape, cache hint
// - tools/call: arguments, results, image content, errors, timeout
// - server/discover and ping (forward compatibility, liveness)
//
// The queue is drained by a helper thread standing in for a host's frame
// callback, because Protocol::HandlePost blocks until the handler has run.
// ============================================================================

#include <gtest/gtest.h>

#include "mcp/CommandQueue.hpp"
#include "mcp/Protocol.hpp"
#include "mcp/ToolRegistry.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <thread>

using namespace quantiloom;
using namespace quantiloom::mcp;
using json = nlohmann::json;

namespace {

/// Runs the queue the way a host would, so a request can complete.
class PumpingHost {
public:
    explicit PumpingHost(CommandQueue& queue) : m_queue(queue) {
        m_thread = std::thread([this] {
            while (!m_stop) {
                m_queue.Drain();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }

    ~PumpingHost() {
        m_stop = true;
        m_thread.join();
    }

private:
    CommandQueue& m_queue;
    std::atomic<bool> m_stop{false};
    std::thread m_thread;
};

class McpProtocolTest : public ::testing::Test {
protected:
    void SetUp() override {
        protocol = std::make_unique<Protocol>(registry, queue, "quantiloom-test", "9.9.9");
    }

    void TearDown() override { queue.Shutdown(); }

    /// Registers a tool that echoes its arguments back.
    void RegisterEcho(const String& name = "ql_echo") {
        ToolDef tool;
        tool.name = name;
        tool.description = "echoes its arguments";
        tool.inputSchemaJson = R"({"type":"object","properties":{"value":{"type":"string"}}})";
        tool.readOnly = true;
        tool.handler = [](const String& args) { return ToolResult::Text(args); };
        ASSERT_TRUE(registry.Add(tool).has_value());
    }

    json Post(const json& request) {
        const Protocol::Response response = protocol->HandlePost(request.dump());
        lastStatus = response.httpStatus;
        if (response.body.empty()) {
            return json();
        }
        return json::parse(response.body);
    }

    ToolRegistry registry;
    CommandQueue queue;
    std::unique_ptr<Protocol> protocol;
    i32 lastStatus = 0;
};

}  // namespace

// ============================================================================
// JSON-RPC framing
// ============================================================================

TEST_F(McpProtocolTest, UnparseableBodyIsAParseError) {
    const Protocol::Response response = protocol->HandlePost("{not json");
    EXPECT_EQ(response.httpStatus, 400);
    const json reply = json::parse(response.body);
    EXPECT_EQ(reply["error"]["code"], -32700);
}

// Batching was removed in 2025-06-18 and never reinstated.
TEST_F(McpProtocolTest, ArrayBodyIsRejected) {
    const Protocol::Response response = protocol->HandlePost(R"([{"jsonrpc":"2.0","id":1}])");
    EXPECT_EQ(response.httpStatus, 400);
    EXPECT_EQ(json::parse(response.body)["error"]["code"], -32600);
}

TEST_F(McpProtocolTest, MissingMethodIsAnInvalidRequest) {
    const json reply = Post({{"jsonrpc", "2.0"}, {"id", 1}});
    EXPECT_EQ(lastStatus, 400);
    EXPECT_EQ(reply["error"]["code"], -32600);
}

// A notification carries no id and gets no reply -- only an acknowledgement.
TEST_F(McpProtocolTest, NotificationIsAcknowledgedWithNoBody) {
    const Protocol::Response response =
        protocol->HandlePost(R"({"jsonrpc":"2.0","method":"notifications/initialized"})");
    EXPECT_EQ(response.httpStatus, 202);
    EXPECT_TRUE(response.body.empty());
}

TEST_F(McpProtocolTest, UnknownMethodIsMethodNotFound) {
    const json reply = Post({{"jsonrpc", "2.0"}, {"id", 7}, {"method", "resources/list"}});
    EXPECT_EQ(reply["error"]["code"], -32601);
    EXPECT_EQ(reply["id"], 7);
}

// ============================================================================
// initialize
// ============================================================================

TEST_F(McpProtocolTest, InitializeEchoesASupportedVersion) {
    const json reply = Post({{"jsonrpc", "2.0"},
                             {"id", 1},
                             {"method", "initialize"},
                             {"params", {{"protocolVersion", "2025-06-18"}}}});
    EXPECT_EQ(reply["result"]["protocolVersion"], "2025-06-18");
}

TEST_F(McpProtocolTest, InitializeFallsBackForAnUnknownVersion) {
    const json reply = Post({{"jsonrpc", "2.0"},
                             {"id", 1},
                             {"method", "initialize"},
                             {"params", {{"protocolVersion", "1999-01-01"}}}});
    EXPECT_EQ(reply["result"]["protocolVersion"], kLatestProtocolVersion);
}

TEST_F(McpProtocolTest, InitializeReportsServerIdentityAndCapabilities) {
    const json reply =
        Post({{"jsonrpc", "2.0"}, {"id", 1}, {"method", "initialize"}, {"params", json::object()}});
    EXPECT_EQ(reply["result"]["serverInfo"]["name"], "quantiloom-test");
    EXPECT_EQ(reply["result"]["serverInfo"]["version"], "9.9.9");
    EXPECT_TRUE(reply["result"]["capabilities"].contains("tools"));
    // Advertising resources or prompts would promise methods that answer -32601.
    EXPECT_FALSE(reply["result"]["capabilities"].contains("resources"));
    EXPECT_FALSE(reply["result"]["capabilities"].contains("prompts"));
}

// ============================================================================
// Catalogue and liveness
// ============================================================================

TEST_F(McpProtocolTest, ToolsListReturnsTheCatalogueWithACacheHint) {
    RegisterEcho();
    const json reply = Post({{"jsonrpc", "2.0"}, {"id", 2}, {"method", "tools/list"}});

    ASSERT_EQ(reply["result"]["tools"].size(), 1u);
    EXPECT_EQ(reply["result"]["tools"][0]["name"], "ql_echo");
    EXPECT_TRUE(reply["result"]["tools"][0]["annotations"]["readOnlyHint"].get<bool>());
    EXPECT_GT(reply["result"]["ttlMs"].get<int>(), 0);
}

TEST_F(McpProtocolTest, PingAnswersEmpty) {
    const json reply = Post({{"jsonrpc", "2.0"}, {"id", 3}, {"method", "ping"}});
    EXPECT_TRUE(reply["result"].is_object());
    EXPECT_TRUE(reply["result"].empty());
}

// 2026-07-28's replacement for the handshake, answered so a newer client can
// see what it is talking to.
TEST_F(McpProtocolTest, DiscoverListsSupportedVersions) {
    const json reply = Post({{"jsonrpc", "2.0"}, {"id", 4}, {"method", "server/discover"}});
    const json versions = reply["result"]["protocolVersions"];
    ASSERT_TRUE(versions.is_array());
    EXPECT_EQ(versions[0], kLatestProtocolVersion);
    EXPECT_EQ(reply["result"]["serverInfo"]["name"], "quantiloom-test");
}

// ============================================================================
// tools/call
// ============================================================================

TEST_F(McpProtocolTest, CallRunsTheHandlerAndReturnsItsText) {
    RegisterEcho();
    const PumpingHost host(queue);

    const json reply = Post({{"jsonrpc", "2.0"},
                             {"id", 5},
                             {"method", "tools/call"},
                             {"params", {{"name", "ql_echo"}, {"arguments", {{"value", "hi"}}}}}});

    ASSERT_TRUE(reply["result"]["content"].is_array());
    EXPECT_EQ(reply["result"]["content"][0]["type"], "text");
    EXPECT_EQ(json::parse(reply["result"]["content"][0]["text"].get<String>())["value"], "hi");
    EXPECT_FALSE(reply["result"]["isError"].get<bool>());
}

TEST_F(McpProtocolTest, CallWithNoArgumentsPassesAnEmptyObject) {
    RegisterEcho();
    const PumpingHost host(queue);

    const json reply = Post(
        {{"jsonrpc", "2.0"}, {"id", 6}, {"method", "tools/call"}, {"params", {{"name", "ql_echo"}}}});

    EXPECT_EQ(reply["result"]["content"][0]["text"], "{}");
}

TEST_F(McpProtocolTest, UnknownToolIsInvalidParams) {
    const json reply = Post({{"jsonrpc", "2.0"},
                             {"id", 7},
                             {"method", "tools/call"},
                             {"params", {{"name", "ql_not_a_tool"}}}});
    EXPECT_EQ(reply["error"]["code"], -32602);
}

TEST_F(McpProtocolTest, MissingToolNameIsInvalidParams) {
    const json reply =
        Post({{"jsonrpc", "2.0"}, {"id", 8}, {"method", "tools/call"}, {"params", json::object()}});
    EXPECT_EQ(reply["error"]["code"], -32602);
}

// A handler that fails reports it in the result, not as a JSON-RPC error: an
// agent can act on "no scene is loaded" and cannot act on a transport fault.
TEST_F(McpProtocolTest, HandlerFailureIsAToolErrorNotAProtocolError) {
    ToolDef tool;
    tool.name = "ql_fails";
    tool.description = "always fails";
    tool.inputSchemaJson = R"({"type":"object"})";
    tool.handler = [](const String&) { return ToolResult::Error("no scene is loaded"); };
    ASSERT_TRUE(registry.Add(tool).has_value());

    const PumpingHost host(queue);
    const json reply = Post({{"jsonrpc", "2.0"},
                             {"id", 9},
                             {"method", "tools/call"},
                             {"params", {{"name", "ql_fails"}}}});

    EXPECT_FALSE(reply.contains("error"));
    EXPECT_TRUE(reply["result"]["isError"].get<bool>());
    EXPECT_EQ(reply["result"]["content"][0]["text"], "no scene is loaded");
}

// Image content must be typed as an image, not pasted into text -- the token
// cost differs by an order of magnitude.
TEST_F(McpProtocolTest, ImageResultBecomesImageContent) {
    ToolDef tool;
    tool.name = "ql_capture";
    tool.description = "returns a picture";
    tool.inputSchemaJson = R"({"type":"object"})";
    tool.handler = [](const String&) {
        ToolResult result;
        result.text = "768x432, 64 samples";
        result.imageBase64 = "iVBORw0KGgo=";
        result.imageMimeType = "image/png";
        return result;
    };
    ASSERT_TRUE(registry.Add(tool).has_value());

    const PumpingHost host(queue);
    const json reply = Post({{"jsonrpc", "2.0"},
                             {"id", 10},
                             {"method", "tools/call"},
                             {"params", {{"name", "ql_capture"}}}});

    const json content = reply["result"]["content"];
    ASSERT_EQ(content.size(), 2u);
    EXPECT_EQ(content[0]["type"], "text");
    EXPECT_EQ(content[1]["type"], "image");
    EXPECT_EQ(content[1]["data"], "iVBORw0KGgo=");
    EXPECT_EQ(content[1]["mimeType"], "image/png");
}

// Nobody drains the queue here, so the call must come back as a readable
// timeout rather than hanging the connection.
TEST_F(McpProtocolTest, CallTimesOutWhenTheHostNeverPumps) {
    ToolDef tool;
    tool.name = "ql_slow";
    tool.description = "never gets run";
    tool.inputSchemaJson = R"({"type":"object"})";
    tool.timeoutMs = 50;
    tool.handler = [](const String&) { return ToolResult::Text("unreachable"); };
    ASSERT_TRUE(registry.Add(tool).has_value());

    const json reply = Post({{"jsonrpc", "2.0"},
                             {"id", 11},
                             {"method", "tools/call"},
                             {"params", {{"name", "ql_slow"}}}});

    EXPECT_TRUE(reply["result"]["isError"].get<bool>());
    EXPECT_NE(reply["result"]["content"][0]["text"].get<String>().find("did not run"),
              String::npos);
}
