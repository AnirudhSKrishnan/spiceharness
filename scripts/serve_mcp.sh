#!/bin/sh
# Absolute-path MCP entry point for desktop clients with a minimal environment.
set -eu
SPICEHARNESS_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PATH="/opt/homebrew/bin:/usr/local/bin:${PATH:-/usr/bin:/bin}"
export PATH
if [ ! -x "$SPICEHARNESS_ROOT/.venv/bin/python" ]; then
    printf '%s\n' "SpiceHarness environment missing. Run: cd '$SPICEHARNESS_ROOT' && uv sync --python 3.13" >&2
    exit 1
fi
exec "$SPICEHARNESS_ROOT/.venv/bin/python" -m spiceharness.mcp_server --workspace "$SPICEHARNESS_ROOT" "$@"
