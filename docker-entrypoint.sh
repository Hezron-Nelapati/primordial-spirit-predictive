#!/usr/bin/env bash
set -e

echo "=== SPSE Predictive startup ==="

# Training and first-run DB build are handled via the web UI.
# Just start Flask and let users interact through the browser.

echo "[ENTRYPOINT] Starting Flask web UI on port ${PORT:-5000}..."
exec python3 webui/server.py
