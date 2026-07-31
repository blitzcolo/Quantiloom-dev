/**
 * @file McpServe.hpp
 * @brief The CLI's MCP mode: a stateless renderer an agent can call
 *
 * Two tools, and nothing that persists between calls. That is not a limitation
 * being worked around -- it is what this host is. OfflineRenderer creates a
 * device, traces one scene and tears the device down again; a configuration
 * arrives, a frame comes back, and the next call starts from nothing. An agent
 * that wants to keep a scene loaded and nudge a light wants Studio, which
 * registers a different set of tools against a context that stays alive.
 *
 * @author blitzcolo
 */

#pragma once

#include "core/Types.hpp"

namespace quantiloom::app {

/**
 * @brief Serve MCP until interrupted
 *
 * @param port                          Loopback port to bind.
 * @param atmosphereModelPackFallback   Passed to every render.
 * @return Process exit code.
 */
int RunMcpServer(u16 port, const String& atmosphereModelPackFallback);

}  // namespace quantiloom::app
