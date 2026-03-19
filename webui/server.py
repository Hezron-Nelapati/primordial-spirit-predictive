"""
Flask web UI server for spse_predictive.
Calls the compiled Rust binary as a subprocess and streams structured output back to the browser.
"""
import os
import subprocess
import re
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Binary path: relative to project root (one level up from webui/)
BINARY = os.environ.get("SPSE_BINARY", os.path.join(os.path.dirname(__file__), "..", "target", "release", "spse_predictive"))


def run_query(query: str, entity: str, domain: str, year: str | None) -> dict:
    cmd = [BINARY, query, entity, domain]
    if year and year.strip():
        cmd.append(year.strip())

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
    except subprocess.TimeoutExpired:
        return {"answer": "Error: query timed out (60 s).", "trace": "", "error": True}
    except FileNotFoundError:
        return {
            "answer": "Error: binary not found. Run `cargo build --release` first.",
            "trace": "",
            "error": True,
        }

    stdout = result.stdout
    stderr = result.stderr

    # Extract [BOT_OUTPUT]: "..." line
    answer = None
    for line in stdout.splitlines():
        m = re.search(r'\[BOT_OUTPUT\]:\s*"(.+)"', line)
        if m:
            answer = m.group(1)

    if answer is None:
        answer = stderr.strip() or "No answer produced."
        return {"answer": answer, "trace": stdout, "error": True}

    return {"answer": answer, "trace": stdout, "error": False}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    q      = (data.get("query") or "").strip()
    entity = (data.get("entity") or "").strip()
    domain = (data.get("domain") or "general").strip()
    year   = (data.get("year") or "").strip() or None

    if not q or not entity:
        return jsonify({"answer": "Query and entity are required.", "trace": "", "error": True}), 400

    return jsonify(run_query(q, entity, domain, year))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
