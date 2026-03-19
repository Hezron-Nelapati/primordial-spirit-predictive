"""
Flask web UI server for SPSE Predictive.

Routes
------
GET  /                  → chat UI
POST /query             → {"query": "..."} — classifies in-process, queries Rust IPC
POST /train/start       → {"passes": N}  — starts training pipeline in background
GET  /train/stream      → SSE stream of training log (param: offset=N)
GET  /train/status      → {running, done, error, log_lines}

Architecture
------------
At startup:
  1. classify_query and minillm_wrapper are imported as modules (models loaded once).
  2. Rust binary is started with --server flag as a persistent subprocess.
  3. Flask waits for "READY" from Rust (DB built + spatial index ready).

Per-query:
  1. classify_query.classify() → intent/tone/domain/entities  (in-process, ~0 ms)
  2. _query_rust() JSON IPC → graph fact                       (persistent process)
  3. minillm_wrapper.style() → conversational answer           (in-process, ~0 ms)
"""
import json, os, re, secrets, subprocess, sys, threading, time

from flask import Flask, Response, jsonify, render_template, request, session

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", secrets.token_hex(32))

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_DIR = os.path.join(ROOT, "python")
BINARY     = os.environ.get(
    "SPSE_BINARY",
    os.path.join(ROOT, "target", "release", "spse_predictive"),
)

# ── In-process Python models ──────────────────────────────────────────────────
# Add python/ to path so we can import classify_query and minillm_wrapper directly.
sys.path.insert(0, PYTHON_DIR)

_classify  = None   # classify_query module
_style_fn  = None   # minillm_wrapper.style function

def _load_python_models():
    """Import and warm up classify_query + minillm_wrapper (called once at startup)."""
    global _classify, _style_fn
    try:
        import classify_query as cq
        cq._get_model()   # pre-load sentence-transformer
        cq._get_nlp()     # pre-load spaCy NER
        _classify = cq
        print("[FLASK] classify_query model loaded.", flush=True)
    except Exception as exc:
        print(f"[FLASK] WARNING: classify_query unavailable ({exc}).", flush=True)

    try:
        import minillm_wrapper as mw
        mw.load_model()   # pre-load SmolLM2 pipeline
        _style_fn = mw.style
        print("[FLASK] minillm_wrapper model loaded.", flush=True)
    except Exception as exc:
        print(f"[FLASK] WARNING: minillm_wrapper unavailable ({exc}).", flush=True)


# ── Persistent Rust process ───────────────────────────────────────────────────
_rust_proc  = None
_rust_lock  = threading.Lock()   # serialize JSON IPC calls (one at a time)
_rust_ready = False

def _start_rust():
    """Start Rust binary with --server flag, wait for READY signal."""
    global _rust_proc, _rust_ready

    if not os.path.isfile(BINARY):
        print(f"[FLASK] WARNING: Rust binary not found at {BINARY}. Queries will fail.", flush=True)
        return

    try:
        _rust_proc = subprocess.Popen(
            [BINARY, "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   # diagnostics go to container logs
            text=True,
            cwd=ROOT,
            bufsize=1,                # line-buffered
        )
        # Forward Rust stderr to our own stdout so it appears in Docker logs.
        def _fwd_stderr():
            for line in iter(_rust_proc.stderr.readline, ""):
                print(f"[RUST] {line}", end="", flush=True)
        threading.Thread(target=_fwd_stderr, daemon=True).start()

        # Wait for READY signal (DB built + spatial index ready).
        print("[FLASK] Waiting for Rust to signal READY...", flush=True)
        deadline = time.time() + 120  # up to 2 min for first-run ingest
        while time.time() < deadline:
            line = _rust_proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line == "READY":
                _rust_ready = True
                print("[FLASK] Rust server ready. System online.", flush=True)
                return
            # Any non-READY line before READY is unexpected — log it.
            print(f"[FLASK/RUST] {line}", flush=True)

        print("[FLASK] WARNING: Rust did not send READY within timeout.", flush=True)
    except Exception as exc:
        print(f"[FLASK] WARNING: Could not start Rust binary ({exc}).", flush=True)


def _query_rust(payload: dict) -> dict:
    """Send a JSON request to the persistent Rust process and return its JSON response."""
    global _rust_proc, _rust_ready

    if not _rust_ready or _rust_proc is None or _rust_proc.poll() is not None:
        return {"answer": "Graph engine not ready. Wait for startup or run Training first.",
                "error": True}

    with _rust_lock:
        try:
            line = json.dumps(payload) + "\n"
            _rust_proc.stdin.write(line)
            _rust_proc.stdin.flush()
            resp_line = _rust_proc.stdout.readline()
            if not resp_line:
                _rust_ready = False
                return {"answer": "Graph engine disconnected.", "error": True}
            return json.loads(resp_line.strip())
        except Exception as exc:
            _rust_ready = False
            return {"answer": f"Graph engine IPC error: {exc}", "error": True}


# ── Session ID ────────────────────────────────────────────────────────────────
def _session_id() -> str:
    if "sid" not in session:
        session["sid"] = secrets.token_hex(8)
    return session["sid"]


# ── Query runner ──────────────────────────────────────────────────────────────
def _run_query(query: str, sid: str) -> dict:
    # Step 1: Classify in-process (intent / tone / domain / entities / NER)
    entity  = "query"
    domain  = "general"
    intent  = "question"
    tone    = "neutral"
    entities: list = []

    if _classify is not None:
        try:
            cls = _classify.classify(query, session_id=sid)
            intent   = cls.get("intent", "question")
            tone     = cls.get("tone",   "neutral")
            domain   = cls.get("domain", "general")
            entities = cls.get("entities", [])
            # Pick entity: first NER entity (preserve original case — Rust graph
            # stores tokens case-sensitively as tokenized from the corpus), or
            # fall back to POS-filtered first noun from the query.
            if entities:
                entity = entities[0]
            else:
                pos_words = _classify._pos_filter(query, _classify.DOMAIN_TAGS).split()
                entity = pos_words[0] if pos_words else "query"
        except Exception as exc:
            print(f"[FLASK] classify failed ({exc}) — using defaults.", flush=True)
    else:
        # Fallback: extract first content word
        stopwords = {"what", "when", "where", "why", "how", "is", "are", "the", "a", "an",
                     "and", "or", "do", "does", "can", "tell", "show", "give"}
        for word in re.sub(r"[^a-z ]", " ", query.lower()).split():
            if word not in stopwords and len(word) > 2:
                entity = word
                break

    # Year from query text (4-digit 1900–2099)
    ym = re.search(r"\b(19|20)\d{2}\b", query)
    year = int(ym.group(0)) if ym else None

    # Step 2: Query Rust via persistent IPC
    payload = {
        "query":      query,
        "entity":     entity,
        "domain":     domain,
        "intent":     intent,
        "tone":       tone,
        "entities":   entities,
        "year":       year,
        "session_id": sid,
    }
    rust_resp = _query_rust(payload)
    graph_fact = rust_resp.get("answer", "")
    is_error   = rust_resp.get("error", True)

    # Step 3: Style through miniLLM in-process (skip on error or System Fault)
    if not is_error and _style_fn is not None and "System Fault" not in graph_fact:
        try:
            graph_fact = _style_fn(graph_fact, query)
        except Exception as exc:
            print(f"[FLASK] miniLLM style failed ({exc}) — using raw fact.", flush=True)

    return {
        "answer": graph_fact,
        "entity": entity,
        "domain": domain,
        "intent": intent,
        "tone":   tone,
        "error":  is_error,
    }


# ── Training state ────────────────────────────────────────────────────────────
_tstate = {"running": False, "done": False, "error": None, "log": []}
_tlock  = threading.Lock()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(force=True) or {}
    q = (data.get("query") or "").strip()
    if not q:
        return jsonify({"answer": "Please enter a message.", "error": True}), 400
    return jsonify(_run_query(q, _session_id()))


# ── Training ──────────────────────────────────────────────────────────────────
@app.route("/train/start", methods=["POST"])
def train_start():
    with _tlock:
        if _tstate["running"]:
            return jsonify({"error": "Training already in progress"}), 400
        data   = request.get_json(force=True) or {}
        passes = max(10, int(data.get("passes", 10)))
        _tstate.update({"running": True, "done": False, "error": None, "log": []})

    pipeline = os.path.join(ROOT, "python", "train_pipeline.py")

    def _run():
        proc = subprocess.Popen(
            [sys.executable, pipeline, "--passes", str(passes)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=ROOT,
        )
        for raw in iter(proc.stdout.readline, ""):
            line = raw.rstrip()
            with _tlock:
                _tstate["log"].append(line)
        proc.wait()
        with _tlock:
            _tstate["running"] = False
            _tstate["done"]    = True
            if proc.returncode != 0:
                _tstate["error"] = f"Pipeline exited with code {proc.returncode}"

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "passes": passes})


@app.route("/train/stream")
def train_stream():
    """SSE: replays all log lines from `offset`, then live-streams until done."""
    offset = int(request.args.get("offset", 0))

    def generate():
        sent = offset
        while True:
            with _tlock:
                log_snap = list(_tstate["log"])
                done     = _tstate["done"]
                error    = _tstate["error"]

            while sent < len(log_snap):
                yield f"data: {json.dumps({'line': log_snap[sent]})}\n\n"
                sent += 1

            if done:
                yield f"data: {json.dumps({'done': True, 'error': error})}\n\n"
                return

            # Keep-alive ping every 0.4 s while training runs
            yield ": ping\n\n"
            time.sleep(0.4)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/train/status")
def train_status():
    with _tlock:
        return jsonify({
            "running":   _tstate["running"],
            "done":      _tstate["done"],
            "error":     _tstate["error"],
            "log_lines": len(_tstate["log"]),
        })


@app.route("/health")
def health():
    return jsonify({"ready": _rust_ready, "rust": _rust_proc is not None})


# ── Startup ───────────────────────────────────────────────────────────────────
def _startup():
    """Called once before Flask starts serving requests."""
    print("[FLASK] Loading Python ML models...", flush=True)
    _load_python_models()
    print("[FLASK] Starting Rust graph engine...", flush=True)
    _start_rust()


if __name__ == "__main__":
    _startup()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
