#!/usr/bin/env python3
"""
Automated training pipeline for SPSE Predictive.

Steps (in order):
  1. Check / download corpus data
  2. Run V2 ingest     → data/v2_corpus.json
  3. Reinforce edges   → duplicate entries N times for heavier edge weights
  4. Run V3 ingest     → data/v3_corpus.json  (skipped if corpus absent)
  5. Train centroids   → data/centroids.json
  6. Reset graph.db    → so Rust rebuilds from reinforced corpus on next query

Progress lines (parsed by Flask SSE stream):
  [STEP]     Major step starting
  [PROGRESS] Within-step detail
  [DONE]     Step completed successfully
  [WARN]     Non-fatal warning
  [ERROR]    Fatal error — script exits with code 1
"""
import argparse, json, os, subprocess, sys

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR  = os.path.join(ROOT, "python")
DATA_DIR    = os.path.join(ROOT, "data")

V2_CORPUS_TXT        = os.path.join(DATA_DIR, "corpus_v2.txt")
V3_CORPUS_TXT        = os.path.join(DATA_DIR, "corpus_v3_massive.txt")
V2_CORPUS_JSON       = os.path.join(DATA_DIR, "v2_corpus.json")
V2_CORPUS_REINFORCED = os.path.join(DATA_DIR, "v2_corpus_reinforced.json")
V3_CORPUS_JSON       = os.path.join(DATA_DIR, "v3_corpus.json")
CENTROIDS_JSON       = os.path.join(DATA_DIR, "centroids.json")
GRAPH_DB             = os.path.join(DATA_DIR, "graph.db")


def log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def run_script(script_name: str) -> int:
    """Run a Python script in python/ directory, streaming all output to stdout.
    Uses cwd=PYTHON_DIR so that '../data/' paths inside scripts resolve correctly
    to the project-root data/ directory."""
    cmd = [sys.executable, os.path.join(PYTHON_DIR, script_name)]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=PYTHON_DIR,
    )
    for line in iter(proc.stdout.readline, ""):
        print(line, end="", flush=True)
    proc.wait()
    return proc.returncode


def reinforce_corpus(src: str, dest: str, passes: int) -> None:
    """Write reinforced corpus to `dest` (src × passes), leaving `src` untouched.
    Writing to a separate file prevents compounding on re-runs: src is always the
    freshly-ingested single-pass output from v2_ingest.py."""
    with open(src, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or passes <= 1:
        # No reinforcement needed — copy src verbatim so Rust always reads dest.
        import shutil
        shutil.copy2(src, dest)
        return
    reinforced = rows * passes
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(reinforced, f)
    log("PROGRESS", f"  {len(rows)} entries × {passes} passes = {len(reinforced)} total")


def main() -> None:
    parser = argparse.ArgumentParser(description="SPSE Predictive training pipeline")
    parser.add_argument(
        "--passes", type=int, default=10,
        help="Ingestion passes for edge reinforcement (minimum 10, can only increase)",
    )
    args = parser.parse_args()
    passes = max(10, args.passes)

    log("INFO", f"Training pipeline starting  (passes={passes})")
    log("INFO", f"Project root: {ROOT}")

    # ── Step 1: Check corpus ──────────────────────────────────────────────────
    log("STEP", "Checking corpus data")

    if not os.path.exists(V2_CORPUS_TXT):
        log("ERROR", "data/corpus_v2.txt not found — V2 corpus is required")
        log("ERROR", "Place a corpus_v2.txt file in the data/ directory and retry.")
        sys.exit(1)
    log("DONE", "V2 corpus (corpus_v2.txt) present")

    v3_present = os.path.exists(V3_CORPUS_TXT)
    if not v3_present:
        log("STEP", "Downloading Wikipedia corpus (corpus_v3_massive.txt) …")
        log("PROGRESS", "  This can take several minutes depending on your connection.")
        rc = run_script("download_corpus.py")
        if rc != 0:
            log("WARN", "Wikipedia download failed — V3 corpus will be skipped")
            v3_present = False
        else:
            v3_present = os.path.exists(V3_CORPUS_TXT)
            if v3_present:
                log("DONE", "Wikipedia corpus downloaded")
            else:
                log("WARN", "Download finished but file not found — skipping V3")
    else:
        log("DONE", "V3 Wikipedia corpus (corpus_v3_massive.txt) already present")

    # ── Step 2: V2 ingest ────────────────────────────────────────────────────
    log("STEP", "Running V2 corpus ingest → data/v2_corpus.json")
    rc = run_script("v2_ingest.py")
    if rc != 0:
        log("ERROR", "V2 ingest failed (see output above)")
        sys.exit(1)
    log("DONE", "V2 ingest complete")

    # ── Step 3: Edge reinforcement ────────────────────────────────────────────
    log("STEP", f"Reinforcing graph edges ({passes}× passes)")
    log("PROGRESS", "  Writing reinforced corpus to v2_corpus_reinforced.json (source unchanged) …")
    try:
        reinforce_corpus(V2_CORPUS_JSON, V2_CORPUS_REINFORCED, passes)
        log("DONE", f"Edge reinforcement complete ({passes}× → {V2_CORPUS_REINFORCED})")
    except Exception as exc:
        log("WARN", f"Edge reinforcement failed ({exc}) — copying source corpus as fallback")
        import shutil
        shutil.copy2(V2_CORPUS_JSON, V2_CORPUS_REINFORCED)

    # ── Step 4: V3 ingest (optional) ─────────────────────────────────────────
    if v3_present:
        log("STEP", "Running V3 Wikipedia ingest → data/v3_corpus.json  (slow)")
        log("PROGRESS", "  This processes the full Wikipedia corpus — may take 10–30 min.")
        rc = run_script("v3_ingest.py")
        if rc != 0:
            log("WARN", "V3 ingest failed — the system will still work without V3 data")
        else:
            log("DONE", "V3 ingest complete")
    else:
        log("STEP", "V3 corpus absent — skipping V3 ingest")

    # ── Step 5: Train centroids ───────────────────────────────────────────────
    log("STEP", "Training centroid classifier → data/centroids.json")
    log("PROGRESS", "  Loading all-MiniLM-L6-v2 and encoding corpus …")
    rc = run_script("train_centroids.py")
    if rc != 0:
        log("ERROR", "Centroid training failed (see output above)")
        sys.exit(1)
    log("DONE", "Centroid training complete")

    # ── Step 6: Reset graph database ─────────────────────────────────────────
    log("STEP", "Resetting graph database")
    if os.path.exists(GRAPH_DB):
        os.remove(GRAPH_DB)
        log("DONE", "graph.db removed — Rust will rebuild from reinforced corpus on next query")
    else:
        log("DONE", "graph.db not present (will be built automatically on first query)")

    log("DONE", "=== Training pipeline complete! The system is ready. ===")


if __name__ == "__main__":
    main()
