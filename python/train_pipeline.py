#!/usr/bin/env python3
"""
Automated training pipeline for SPSE Predictive.

Steps (in order):
  0. Bootstrap centroids → data/centroids.json       (reads corpus.txt + mock labels, ~98k sentences)
  1. Check / download corpus data
  2. Run ingest          → data/corpus_tmp.json       (uses step-0 centroids ✓)
  3. Run ingest_wiki     → data/corpus_wiki_tmp.json  (uses step-0 centroids ✓, optional)
  4. Merge all           → data/corpus.json            (unified, with real centroid labels)
  5. Reinforce edges     → data/corpus_reinforced.json
  6. Retrain centroids   → data/centroids.json         (now from full centroid-labelled corpus)
  7. Reset graph.db      → so Rust rebuilds from reinforced corpus on next query

Why step 0 reads corpus.txt directly:
  ingest.py classifies every sentence using centroids.json.  On a first run
  centroids.json does not exist.  train_centroids.py now reads corpus.txt and
  labels sentences with mock_classify() to produce initial centroids from 98k
  real sentences — far better than 15 hardcoded seed phrases.  Step 6 retrains
  from the corpus.json produced by ingest (which used the step-0 centroids),
  completing one full refinement cycle.

Progress lines (parsed by Flask SSE stream):
  [STEP]     Major step starting
  [PROGRESS] Within-step detail
  [DONE]     Step completed successfully
  [WARN]     Non-fatal warning
  [ERROR]    Fatal error — script exits with code 1
"""
import argparse, json, os, shutil, subprocess, sys

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR  = os.path.join(ROOT, "python")
DATA_DIR    = os.path.join(ROOT, "data")

BASE_CORPUS_TXT     = os.path.join(DATA_DIR, "corpus.txt")
WIKI_CORPUS_TXT     = os.path.join(DATA_DIR, "corpus_wiki.txt")
BASE_TMP_JSON       = os.path.join(DATA_DIR, "corpus_tmp.json")       # intermediate
WIKI_TMP_JSON       = os.path.join(DATA_DIR, "corpus_wiki_tmp.json")  # intermediate
CORPUS_JSON         = os.path.join(DATA_DIR, "corpus.json")           # merged output
CORPUS_REINFORCED   = os.path.join(DATA_DIR, "corpus_reinforced.json")
CENTROIDS_JSON      = os.path.join(DATA_DIR, "centroids.json")
GRAPH_DB            = os.path.join(DATA_DIR, "graph.db")


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
    freshly-merged single-pass output."""
    with open(src, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or passes <= 1:
        # No reinforcement needed — copy src verbatim so Rust always reads dest.
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

    # ── Step 0: Bootstrap centroids from corpus.txt ──────────────────────────
    # train_centroids.py reads corpus.txt directly and labels sentences with
    # mock_classify() — producing centroids from ~98k real sentences rather
    # than 15 seed phrases.  Skipped on re-runs where centroids.json exists.
    if not os.path.exists(CENTROIDS_JSON):
        log("STEP", "Bootstrapping centroids from corpus.txt (mock labels, ~98k sentences)")
        rc = run_script("train_centroids.py")
        if rc != 0:
            log("WARN", "Bootstrap centroid training failed — ingest will use keyword fallback")
        else:
            log("DONE", "Bootstrap centroids ready → data/centroids.json")
    else:
        log("STEP", "centroids.json present — skipping bootstrap")

    # ── Step 1: Check corpus ──────────────────────────────────────────────────
    log("STEP", "Checking corpus data")

    if not os.path.exists(BASE_CORPUS_TXT):
        log("ERROR", "data/corpus.txt not found — base corpus is required")
        log("ERROR", "Place a corpus.txt file in the data/ directory and retry.")
        sys.exit(1)
    log("DONE", "Base corpus (corpus.txt) present")

    wiki_present = os.path.exists(WIKI_CORPUS_TXT)
    if not wiki_present:
        log("STEP", "Downloading Wikipedia corpus (corpus_wiki.txt) …")
        log("PROGRESS", "  This can take several minutes depending on your connection.")
        rc = run_script("download_corpus.py")
        if rc != 0:
            log("WARN", "Wikipedia download failed — Wikipedia corpus will be skipped")
            wiki_present = False
        else:
            wiki_present = os.path.exists(WIKI_CORPUS_TXT)
            if wiki_present:
                log("DONE", "Wikipedia corpus downloaded")
            else:
                log("WARN", "Download finished but file not found — skipping Wikipedia ingest")
    else:
        log("DONE", "Wikipedia corpus (corpus_wiki.txt) already present")

    # ── Step 2: Base corpus ingest ────────────────────────────────────────────
    log("STEP", "Running base corpus ingest")
    rc = run_script("ingest.py")
    if rc != 0:
        log("ERROR", "Base corpus ingest failed (see output above)")
        sys.exit(1)
    log("DONE", "Base corpus ingest complete")

    # ── Step 3: Wikipedia ingest (optional) ───────────────────────────────────
    if wiki_present:
        log("STEP", "Running Wikipedia ingest (slow)")
        log("PROGRESS", "  This processes the full Wikipedia corpus — may take 10–30 min.")
        rc = run_script("ingest_wiki.py")
        if rc != 0:
            log("WARN", "Wikipedia ingest failed — the system will still work without Wikipedia data")
            wiki_present = False
        else:
            log("DONE", "Wikipedia ingest complete")
    else:
        log("STEP", "Wikipedia corpus absent — skipping Wikipedia ingest")

    # ── Step 4: Merge into unified corpus.json ───────────────────────────────
    log("STEP", "Merging ingested corpora → data/corpus.json")
    merged_data = []
    try:
        with open(BASE_TMP_JSON, encoding="utf-8") as f:
            merged_data.extend(json.load(f))
        log("PROGRESS", f"  Added {len(merged_data)} entries from base corpus ingest.")
    except FileNotFoundError:
        log("ERROR", f"Base ingest output not found: {BASE_TMP_JSON}")
        sys.exit(1)

    if wiki_present:
        try:
            with open(WIKI_TMP_JSON, encoding="utf-8") as f:
                wiki_data = json.load(f)
                merged_data.extend(wiki_data)
            log("PROGRESS", f"  Added {len(wiki_data)} entries from Wikipedia ingest. Total: {len(merged_data)}")
        except FileNotFoundError:
            log("WARN", f"Wikipedia ingest output not found: {WIKI_TMP_JSON} — skipping.")
        except json.JSONDecodeError:
            log("WARN", f"Wikipedia ingest output is not valid JSON — skipping.")

    with open(CORPUS_JSON, "w", encoding="utf-8") as f:
        json.dump(merged_data, f)
    log("DONE", f"Unified corpus created: {len(merged_data)} entries → data/corpus.json")

    # Clean up intermediate ingest files
    for tmp in (BASE_TMP_JSON, WIKI_TMP_JSON):
        if os.path.exists(tmp):
            os.remove(tmp)

    # ── Step 5: Edge reinforcement ────────────────────────────────────────────
    log("STEP", f"Reinforcing graph edges ({passes}× passes)")
    log("PROGRESS", "  Writing reinforced corpus to corpus_reinforced.json (source unchanged) …")
    try:
        reinforce_corpus(CORPUS_JSON, CORPUS_REINFORCED, passes)
        log("DONE", f"Edge reinforcement complete ({passes}× → data/corpus_reinforced.json)")
    except Exception as exc:
        log("WARN", f"Edge reinforcement failed ({exc}) — copying source corpus as fallback")
        shutil.copy2(CORPUS_JSON, CORPUS_REINFORCED)

    # ── Step 6: Train centroids ───────────────────────────────────────────────
    log("STEP", "Training centroid classifier → data/centroids.json")
    log("PROGRESS", "  Loading sentence-transformer model and encoding corpus …")
    rc = run_script("train_centroids.py")
    if rc != 0:
        log("ERROR", "Centroid training failed (see output above)")
        sys.exit(1)
    log("DONE", "Centroid training complete")

    # ── Step 7: Reset graph database ─────────────────────────────────────────
    # Rust reads corpus_reinforced.json (or corpus.json as fallback) at startup
    # when graph.db is absent.  Removing the DB forces a fresh ingest.
    log("STEP", "Resetting graph database")
    if os.path.exists(GRAPH_DB):
        os.remove(GRAPH_DB)
        log("DONE", "graph.db removed — Rust will rebuild from reinforced corpus on next query")
    else:
        log("DONE", "graph.db not present (will be built automatically on first query)")

    log("DONE", "=== Training pipeline complete! The system is ready. ===")


if __name__ == "__main__":
    main()
