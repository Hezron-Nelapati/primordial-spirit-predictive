"""
Flask web UI server for SPSE Predictive.

Routes
------
GET  /                  → chat UI
POST /query             → {"query": "..."} — auto-extracts entity/domain
POST /train/start       → {"passes": N}  — starts training pipeline in background
GET  /train/stream      → SSE stream of training log (param: offset=N)
GET  /train/status      → {running, done, error, log_lines}
"""
import json, os, re, secrets, subprocess, sys, threading

from flask import Flask, Response, jsonify, render_template, request, session

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", secrets.token_hex(32))

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BINARY = os.environ.get(
    "SPSE_BINARY",
    os.path.join(ROOT, "target", "release", "spse_predictive"),
)

# ── Training state ────────────────────────────────────────────────────────────
_tstate = {"running": False, "done": False, "error": None, "log": []}
_tlock  = threading.Lock()

# ── Entity / domain inference ─────────────────────────────────────────────────
_STOPWORDS = {
    "what", "when", "where", "why", "how", "is", "are", "was", "were", "the",
    "a", "an", "and", "or", "but", "do", "does", "did", "can", "could",
    "would", "should", "will", "have", "has", "had", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "tell", "show", "give", "make",
    "get", "let", "go", "who", "which", "that", "this", "these", "those",
    "not", "no", "please", "about", "from", "into", "over", "under",
    "between", "explain", "describe", "define", "find", "list",
}

_DOMAIN_KEYWORDS = {
    "tech":      {"server", "cpu", "api", "network", "router", "code", "software",
                  "hardware", "database", "computer", "program", "bug", "memory",
                  "disk", "online", "internet", "cloud", "algorithm", "cache",
                  "kernel", "binary", "compiler", "runtime"},
    "finance":   {"bank", "money", "loan", "credit", "interest", "investment",
                  "stock", "market", "cash", "fund", "atm", "currency", "price",
                  "cost", "budget", "revenue", "profit", "tax", "asset", "liability",
                  "dividend", "inflation", "equity"},
    "science":   {"quantum", "physics", "chemistry", "biology", "atom", "molecule",
                  "energy", "force", "mass", "particle", "gravity", "light", "wave",
                  "electron", "proton", "experiment", "theory", "hypothesis",
                  "equation", "nucleus", "velocity"},
    "geography": {"city", "country", "capital", "continent", "ocean", "river",
                  "mountain", "paris", "london", "amazon", "europe", "africa",
                  "asia", "america", "climate", "geography", "region", "nation",
                  "lake", "desert", "island"},
}


def _extract_entity(query: str) -> str:
    """Extract the most semantically important word using NLTK POS tagging."""
    try:
        import nltk
        # Ensure punkt available (silent; already downloaded in Docker)
        tokens = nltk.word_tokenize(query.lower())
        tagged = nltk.pos_tag(tokens)
        # Prefer proper noun → common noun, skip stopwords and short tokens
        for tag_group in [("NNP", "NNPS"), ("NN", "NNS")]:
            for word, tag in tagged:
                if (tag in tag_group
                        and word not in _STOPWORDS
                        and word.isalpha()
                        and len(word) > 2):
                    return word
    except Exception:
        pass
    # Fallback: first content word (no ML required)
    for word in query.lower().split():
        w = re.sub(r"[^a-z]", "", word)
        if w and w not in _STOPWORDS and len(w) > 2:
            return w
    return re.sub(r"[^a-z]", "", query.split()[0].lower()) if query.split() else "query"


def _infer_domain(query: str) -> str:
    words = set(re.sub(r"[^a-z ]", " ", query.lower()).split())
    for domain, kws in _DOMAIN_KEYWORDS.items():
        if words & kws:
            return domain
    return "general"


# ── Session ID ────────────────────────────────────────────────────────────────
def _session_id() -> str:
    if "sid" not in session:
        session["sid"] = secrets.token_hex(8)
    return session["sid"]


# ── Query runner ──────────────────────────────────────────────────────────────
def _run_query(query: str, sid: str) -> dict:
    entity = _extract_entity(query)
    domain = _infer_domain(query)

    # Year from query text (4-digit 1900–2099)
    ym = re.search(r"\b(19|20)\d{2}\b", query)
    year = ym.group(0) if ym else None

    cmd = [BINARY, query, entity, domain]
    if year:
        cmd.append(year)
    cmd += ["--session-id", sid]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {
            "answer": "Query timed out (60 s). The graph may still be building.",
            "trace": "", "entity": entity, "domain": domain, "error": True,
        }
    except FileNotFoundError:
        return {
            "answer": (
                "Binary not found. "
                "Run `cargo build --release` locally, or rebuild the Docker image."
            ),
            "trace": "", "entity": entity, "domain": domain, "error": True,
        }

    stdout = result.stdout
    answer = None
    for line in stdout.splitlines():
        m = re.search(r'\[BOT_OUTPUT\]:\s*"(.+)"', line)
        if m:
            answer = m.group(1)
            break

    if answer is None:
        fallback = result.stderr.strip() or "No answer produced — try running Training first."
        return {"answer": fallback, "trace": stdout, "entity": entity, "domain": domain, "error": True}

    return {"answer": answer, "trace": stdout, "entity": entity, "domain": domain, "error": False}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(force=True) or {}
    q = (data.get("query") or "").strip()
    if not q:
        return jsonify({"answer": "Please enter a message.", "trace": "",
                        "entity": "", "domain": "", "error": True}), 400
    return jsonify(_run_query(q, _session_id()))


# ── Training ──────────────────────────────────────────────────────────────────
@app.route("/train/start", methods=["POST"])
def train_start():
    with _tlock:
        if _tstate["running"]:
            return jsonify({"error": "Training already in progress"}), 400
        data  = request.get_json(force=True) or {}
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
        import time
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
