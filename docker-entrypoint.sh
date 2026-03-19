#!/usr/bin/env bash
set -e

echo "=== SPSE Predictive startup ==="

# ── 1. Ensure corpus files exist ─────────────────────────────────────────────
if [ ! -f "data/v2_corpus.json" ]; then
  echo "[ENTRYPOINT] data/v2_corpus.json not found."
  echo "[ENTRYPOINT] Run the ingest scripts from the python/ directory first:"
  echo "             cd python && python3 v2_ingest.py"
  echo "[ENTRYPOINT] Continuing anyway — first query will fail if DB is empty."
fi

# ── 2. Warm-up query: triggers first-run DB ingest if graph.db is absent ─────
if [ ! -f "data/graph.db" ] && [ -f "data/v2_corpus.json" ]; then
  echo "[ENTRYPOINT] graph.db absent — running warm-up query to build DB..."
  ./spse_predictive "Is the server online?" server tech 2026 > /dev/null 2>&1 || true
  echo "[ENTRYPOINT] DB warm-up complete."
fi

# ── 3. Train centroids if not present ────────────────────────────────────────
if [ ! -f "data/centroids.json" ]; then
  echo "[ENTRYPOINT] centroids.json absent — training centroids..."
  (cd python && python3 train_centroids.py) || echo "[ENTRYPOINT] Centroid training failed — classification will use fallback."
fi

# ── 4. Start Flask server ─────────────────────────────────────────────────────
echo "[ENTRYPOINT] Starting Flask web UI on port ${PORT:-5000}..."
exec python3 webui/server.py
