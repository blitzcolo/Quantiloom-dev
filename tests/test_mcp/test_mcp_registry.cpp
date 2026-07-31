// ============================================================================
// Quantiloom - Unit Tests for mcp/ToolRegistry.hpp and mcp/CommandQueue.hpp
// ============================================================================
// Tests cover:
// - Tool registration: validation, replacement, JSON schema parsing
// - Catalogue listing: ordering, annotations
// - Command queue: host-thread execution, ordering, timeout, wake callback,
//   host dispatch routing, shutdown releasing waiters
// ============================================================================

#include <gtest/gtest.h>

#include "mcp/CommandQueue.hpp"
#include "mcp/ToolRegistry.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <stdexcept>
#include <thread>

using namespace quantiloom;
using namespace quantiloom::mcp;
using json = nlohmann::json;

namespace {

ToolDef MakeTool(const String& name, const bool readOnly = true) {
    ToolDef tool;
    tool.name = name;
    tool.description = "a test tool";
    tool.inputSchemaJson = R"({"type":"object","properties":{}})";
    tool.readOnly = readOnly;
    tool.handler = [](const String&) { return ToolResult::Text("ok"); };
    return tool;
}

}  // namespace

// ============================================================================
// ToolRegistry
// ============================================================================

TEST(McpToolRegistry, AddsAndFindsATool) {
    ToolRegistry registry;
    ASSERT_TRUE(registry.Add(MakeTool("ql_get_status")).has_value());

    EXPECT_EQ(registry.Size(), 1u);
    const auto found = registry.Find("ql_get_status");
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->name, "ql_get_status");
}

TEST(McpToolRegistry, FindReturnsNulloptForUnknownName) {
    ToolRegistry registry;
    EXPECT_FALSE(registry.Find("ql_nonexistent").has_value());
}

TEST(McpToolRegistry, RejectsEmptyName) {
    ToolRegistry registry;
    ToolDef tool = MakeTool("");
    EXPECT_FALSE(registry.Add(tool).has_value());
}

TEST(McpToolRegistry, RejectsMissingHandler) {
    ToolRegistry registry;
    ToolDef tool = MakeTool("ql_no_handler");
    tool.handler = nullptr;
    EXPECT_FALSE(registry.Add(tool).has_value());
}

// A schema that only fails to parse when a client asks for the catalogue would
// take the whole catalogue down with it, so it is rejected at registration.
TEST(McpToolRegistry, RejectsUnparseableSchema) {
    ToolRegistry registry;
    ToolDef tool = MakeTool("ql_bad_schema");
    tool.inputSchemaJson = "{ this is not json";
    EXPECT_FALSE(registry.Add(tool).has_value());
}

TEST(McpToolRegistry, RejectsNonObjectSchema) {
    ToolRegistry registry;
    ToolDef tool = MakeTool("ql_array_schema");
    tool.inputSchemaJson = R"(["not","an","object"])";
    EXPECT_FALSE(registry.Add(tool).has_value());
}

TEST(McpToolRegistry, RegisteringTwiceReplaces) {
    ToolRegistry registry;
    ASSERT_TRUE(registry.Add(MakeTool("ql_thing")).has_value());

    ToolDef replacement = MakeTool("ql_thing");
    replacement.description = "the second one";
    ASSERT_TRUE(registry.Add(replacement).has_value());

    EXPECT_EQ(registry.Size(), 1u);
    EXPECT_EQ(registry.Find("ql_thing")->description, "the second one");
}

// A client may cache the catalogue, so the order has to be the same every time.
TEST(McpToolRegistry, ListIsOrderedByName) {
    ToolRegistry registry;
    ASSERT_TRUE(registry.Add(MakeTool("ql_zebra")).has_value());
    ASSERT_TRUE(registry.Add(MakeTool("ql_alpha")).has_value());
    ASSERT_TRUE(registry.Add(MakeTool("ql_middle")).has_value());

    const json list = json::parse(registry.ListJson());
    ASSERT_EQ(list.size(), 3u);
    EXPECT_EQ(list[0]["name"], "ql_alpha");
    EXPECT_EQ(list[1]["name"], "ql_middle");
    EXPECT_EQ(list[2]["name"], "ql_zebra");
}

TEST(McpToolRegistry, ListCarriesSchemaAndAnnotations) {
    ToolRegistry registry;
    ToolDef tool = MakeTool("ql_delete_thing", /*readOnly=*/false);
    tool.destructive = true;
    tool.inputSchemaJson = R"({"type":"object","properties":{"index":{"type":"integer"}}})";
    ASSERT_TRUE(registry.Add(tool).has_value());

    const json entry = json::parse(registry.ListJson())[0];
    EXPECT_EQ(entry["inputSchema"]["properties"]["index"]["type"], "integer");
    EXPECT_FALSE(entry["annotations"]["readOnlyHint"].get<bool>());
    EXPECT_TRUE(entry["annotations"]["destructiveHint"].get<bool>());
}

// ============================================================================
// CommandQueue
// ============================================================================

TEST(McpCommandQueue, WorkRunsOnTheDrainingThread) {
    CommandQueue queue;
    std::atomic<bool> ran{false};
    std::thread::id ranOn;

    std::thread submitter([&] {
        queue.Submit([&] {
            ranOn = std::this_thread::get_id();
            ran = true;
        }, false, 2000);
    });

    // Stand in for the host's frame callback.
    while (!ran) {
        queue.Drain();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    submitter.join();

    EXPECT_TRUE(ran);
    EXPECT_EQ(ranOn, std::this_thread::get_id());
}

TEST(McpCommandQueue, SubmitReturnsTrueOnceDrained) {
    CommandQueue queue;
    std::atomic<bool> result{false};
    std::atomic<bool> finished{false};

    std::thread submitter([&] {
        result = queue.Submit([] {}, false, 2000);
        finished = true;
    });

    while (!finished) {
        queue.Drain();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    submitter.join();

    EXPECT_TRUE(result);
}

// A host that never pumps must produce a timeout the agent can read, not a
// connection that hangs.
TEST(McpCommandQueue, TimesOutWhenNobodyDrains) {
    CommandQueue queue;
    const auto start = std::chrono::steady_clock::now();
    const bool ran = queue.Submit([] {}, false, 50);
    const auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(ran);
    EXPECT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 45);
}

TEST(McpCommandQueue, DrainRunsInSubmissionOrder) {
    CommandQueue queue;
    Vector<int> order;
    std::atomic<int> done{0};

    Vector<std::thread> submitters;
    // Submitted one at a time so the order is defined; the point under test is
    // that Drain preserves it, not that concurrent submits sort themselves out.
    for (u32 i = 0; i < 3; ++i) {
        submitters.emplace_back([&queue, &order, &done, i] {
            queue.Submit([&order, i] { order.push_back(static_cast<int>(i)); }, false, 2000);
            ++done;
        });
        while (queue.Pending() <= i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    while (done < 3) {
        queue.Drain();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    for (auto& t : submitters) {
        t.join();
    }

    ASSERT_EQ(order.size(), 3u);
    EXPECT_EQ(order[0], 0);
    EXPECT_EQ(order[1], 1);
    EXPECT_EQ(order[2], 2);
}

// Studio stops requesting frames when the viewport is paused, so without this
// nudge a queued call would wait out its whole timeout.
TEST(McpCommandQueue, WakeCallbackFiresOnSubmit) {
    CommandQueue queue;
    std::atomic<int> wakes{0};
    queue.SetWakeCallback([&] { ++wakes; });

    std::thread submitter([&] { queue.Submit([] {}, false, 200); });
    while (wakes == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    queue.Drain();
    submitter.join();

    EXPECT_GE(wakes.load(), 1);
}

TEST(McpCommandQueue, HostDispatchWorkIsHandedOverNotRun) {
    CommandQueue queue;
    Vector<std::function<void()>> deferred;
    queue.SetHostDispatch([&](std::function<void()> work) { deferred.push_back(std::move(work)); });

    std::atomic<bool> ran{false};
    std::thread submitter([&] { queue.Submit([&] { ran = true; }, true, 2000); });

    while (queue.Pending() == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    queue.Drain();

    // Drain handed it over; nothing has run it yet.
    EXPECT_EQ(deferred.size(), 1u);
    EXPECT_FALSE(ran);

    deferred[0]();  // the host's event loop gets to it
    submitter.join();
    EXPECT_TRUE(ran);
}

TEST(McpCommandQueue, HostDispatchRunsInlineWhenNoDispatcherIsSet) {
    CommandQueue queue;
    std::atomic<bool> ran{false};
    std::thread submitter([&] { queue.Submit([&] { ran = true; }, true, 2000); });

    while (!ran) {
        queue.Drain();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    submitter.join();
    EXPECT_TRUE(ran);
}

// The host is going away: waiters are released, but their handlers must not run
// against state that is being torn down.
TEST(McpCommandQueue, ShutdownReleasesWaitersWithoutRunningThem) {
    CommandQueue queue;
    std::atomic<bool> ran{false};
    std::atomic<bool> result{true};
    std::atomic<bool> finished{false};

    std::thread submitter([&] {
        result = queue.Submit([&] { ran = true; }, false, 5000);
        finished = true;
    });

    while (queue.Pending() == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    queue.Shutdown();
    submitter.join();

    EXPECT_TRUE(finished);
    EXPECT_FALSE(ran);
    EXPECT_FALSE(result);
}

TEST(McpCommandQueue, SubmitAfterShutdownFailsImmediately) {
    CommandQueue queue;
    queue.Shutdown();
    EXPECT_FALSE(queue.Submit([] {}, false, 5000));
}

// A throwing handler must not take the host's frame callback with it.
TEST(McpCommandQueue, HandlerExceptionIsContained) {
    CommandQueue queue;
    std::atomic<bool> finished{false};
    std::atomic<bool> result{false};

    std::thread submitter([&] {
        result = queue.Submit([] { throw std::runtime_error("boom"); }, false, 2000);
        finished = true;
    });

    while (!finished) {
        EXPECT_NO_THROW(queue.Drain());
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    submitter.join();

    EXPECT_TRUE(result);
}
