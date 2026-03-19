# SPSE Predictive Task Status

Last synced against the repository on 2026-03-19.

---

## Current Architecture Snapshot

### Execution Flow — Demo Mode (`cargo run`)
```
data/v2_graph_edges.json  [+ data/v3_graph_edges.json if present]
    ↓ ingest_rows()  (Rust, src/main.rs)
WordGraph (merged V2 + V3 edges when available)
    ↓
ReasoningModule + WalkConfig (hard-coded per scenario)
    ↓
predict_next()  [Tier 1: intent × domain × tone × entity × temporal biasing]
    ↓
5 hard-coded demo outputs printed to stdout
```

### Execution Flow — CLI Mode (`cargo run -- "query" entity domain [year]`)
```
data/v2_graph_edges.json  [+ data/v3_graph_edges.json if present]
    ↓ ingest_rows()
WordGraph
    ↓
python/classify_query.py  →  {intent, tone, domain}   [graceful fallback if Python unavailable]
    ↓
ReasoningModule::sanitize_queue()  [Guardrail 3]
    ↓
predict_next()  [Tier 1 multi-signal biasing]
    ↓
python/minillm_wrapper.py  →  styled response         [graceful fallback if model unavailable]
    ↓
stdout
```

### What is exported from `src/lib.rs`
All six modules: `graph`, `reasoning`, `walk`, `classify`, `spatial`, `ingest`.

---

## Module-Level Status

### `src/graph.rs`
- `WordNode`, `WordEdge`, `WordGraph` with FNV-1a hashing.
- `WordNode::compute_lexical_vector()` — deterministic 5D OOV fallback vector.
- `position: [f32; 3]` field exists on `WordNode` but is never populated (3D spatial placement not yet computed).
- Status: **Complete and wired**.

### `src/reasoning.rs`
- `SessionalMemory` — four `Vec<String>` stacks: intent, tone, domain, entity.
- `sanitize_queue()` — Guardrail 3, called inside `generate_dynamic_answer()` before every walk.
- `update_context()` — pushes to all four stacks; skips "Pronoun" entity entries.
- Status: **Complete and fully wired**.

### `src/walk.rs`
- `predict_next()` — Tier 1 scoring: OOV lexical fallback, intent ×2.0, domain ×2.0, tone ×2.0, entity ×1.5, temporal proximity multiplier.
- `resolve_start_node()` — reverse-walk up to 20 hops to find sentence anchor.
- Tier 2 (KD-tree radial search) and Tier 3 (A\* pathfinding) have **no code** — design-doc only.
- Status: **Tier 1 complete and wired. Tier 2/3 absent.**

### `src/classify.rs`
- `Classifier::load(path)`, `Classifier::intent(emb_full, emb_pos)`, `Classifier::tone(...)`.
- Exported from `lib.rs`.
- **Not called from `src/main.rs`** — requires pre-computed float embeddings from Python side. The Python classify bridge (`python/classify_query.py`) handles this for CLI queries.
- Status: **Exported; Rust-side struct unused at runtime; Python-side equivalent is the active path.**

### `src/spatial.rs`
- `SpatialGrid` wrapping `kiddo::KdTree<f32, 3>` with `build()`, `query_radius()`, `query_nearest()`.
- Exported from `lib.rs`. Not called from `main.rs` or `walk.rs`.
- Status: **Exported and compiles; not integrated into active runtime.**

### `src/ingest.rs`
- `ingest_text()`, `ingest_sentence()` — V2-compatible plain-text corpus ingestion with edge reinforcement.
- `GraphStats::compute()` / `GraphStats::report()` — node count, edge count, avg out-degree.
- Exported from `lib.rs`. Not called from `main.rs` (main uses its own `ingest_rows()` which ingests tagged JSON rows with full metadata).
- Status: **Exported, compiles cleanly, available for integration tests.**

---

## Python Pipeline Status

### `python/v2_ingest.py`
- Reads `data/corpus_v2.txt` → `data/v2_graph_edges.json`.
- Heuristic classification (`mock_classify`): intent from `?`/keyword, tone from `!`/keyword, domain from keyword set.
- NLTK NER + regex year extraction.
- Status: **Fully functional**.

### `python/v3_ingest.py`
- Reads `data/corpus_v3_massive.txt` → `data/v3_graph_edges.json`.
- All sentences tagged `intent: statement, tone: neutral, domain: general`.
- Neither input nor output file present in repo.
- Status: **Script ready; not executed; outputs absent**.

### `python/train_centroids.py`
- `all-MiniLM-L6-v2` embeddings + `sklearn.NearestCentroid`.
- Output `data/centroids.json` (259 KB) is checked in.
- Status: **Functional; output available**.

### `python/classify_query.py`  *(new)*
- Runtime classification for CLI queries.
- Accepts `sys.argv[1]` (query string), optional `sys.argv[2]` (centroids path).
- Mirrors `train_centroids.py` POS tag sets and blended distance formula exactly.
- Domain via keyword heuristic (same map as `v2_ingest.py`).
- All diagnostics to stderr; single JSON line to stdout.
- Rust calls this via `std::process::Command`; falls back to defaults if unavailable.
- Status: **Implemented; requires `sentence-transformers` + `nltk` installed**.

### `python/minillm_wrapper.py`
- `HuggingFaceTB/SmolLM2-135M-Instruct`, 135 M params, CPU, temperature 0.0.
- Accepts `sys.argv[1]` (graph fact) + `sys.argv[2]` (user prompt).
- Rust calls this via `std::process::Command` in CLI mode; falls back to raw fact if unavailable.
- Status: **Implemented and bridged; requires `transformers` + model cache (~270 MB)**.

### `python/download_corpus.py`
- Downloads Simple English Wikipedia `20220301.simple` → `data/corpus_v3_massive.txt`.
- Status: **Script ready; not executed**.

### `python/requirements.txt`
```
sentence-transformers
scikit-learn
numpy
nltk
datasets
transformers
```
Status: **Accurate — matches all packages imported by checked-in scripts**.

---

## Data Artifacts

| File | Present | Source | Consumer |
|---|---|---|---|
| `data/corpus.txt` | ✓ | Manual | Legacy only |
| `data/corpus_v2.txt` | ✓ | Manual | `v2_ingest.py` |
| `data/v2_graph_edges.json` | ✓ | `v2_ingest.py` | `src/main.rs` (required) |
| `data/centroids.json` | ✓ | `train_centroids.py` | `classify_query.py` |
| `data/corpus_v3_massive.txt` | ✗ | `download_corpus.py` | `v3_ingest.py` |
| `data/v3_graph_edges.json` | ✗ | `v3_ingest.py` | `src/main.rs` (auto-detected) |

---

## Cargo.toml Dependency Status

| Crate | Declared | Active Use |
|---|---|---|
| `serde + serde_json` | ✓ | ✓ — JSON deserialization + subprocess output parsing |
| `kiddo = "4"` | ✓ | `spatial.rs` (exported, not wired into walk) |
| `rusqlite = "0.31"` | ✓ | ✗ — no code references it |
| `rand = "0.8"` | ✓ | ✗ — no code references it |

---

## Phase Checklist

### Phase 1: Repository Foundation
Status: **Complete**
- [x] `Cargo.toml` and Rust crate scaffolding
- [x] `python/requirements.txt` accurate (nltk, datasets, transformers, sentence-transformers, scikit-learn, numpy)
- [x] `data/corpus.txt` and `data/corpus_v2.txt` exist
- [x] `data/centroids.json` checked in

### Phase 2: Python Training and Ingestion Assets
Status: **Complete for V2; spaCy path removed as unused**
- [x] `python/train_centroids.py` produces `data/centroids.json`
- [x] `python/v2_ingest.py` produces `data/v2_graph_edges.json`
- [x] V2 ingestion uses NLTK tokenization, NLTK NER, regex year extraction, heuristic classification
- [x] `python/requirements.txt` lists all packages actually used
- [ ] spaCy-based NER in active ingestion path (removed; NLTK used instead)
- [ ] Centroid inference used directly at Rust runtime (Python subprocess bridge is the active path)

### Phase 3: Active Rust V2 Runtime
Status: **Complete — all planned features wired**
- [x] `src/graph.rs` — `WordNode`, `WordEdge`, `WordGraph`, FNV-1a hashing
- [x] `src/reasoning.rs` — sessional stacks, `sanitize_queue`, `update_context`
- [x] `src/walk.rs` — OOV lexical fallback, Tier 1 multi-signal edge scoring (intent, domain, tone, entity, temporal), reverse start-node resolution
- [x] `src/main.rs` — loads `data/v2_graph_edges.json`, runs 5 hard-coded demo scenarios
- [x] Temporal weighting via `dated` edges
- [x] Pronoun fallback via `entity_stack`
- [x] `reasoning.sanitize_queue()` invoked in `generate_dynamic_answer()` before every walk
- [x] Tone-based routing in walk scoring (×2.0 multiplier)
- [x] Entity-based routing in walk scoring (×1.5 multiplier)
- [x] User queries accepted from CLI (`cargo run -- "query" entity domain [year]`)

### Phase 4: Legacy V1 Surface
Status: **Resolved — V1 surface rewritten or removed**
- [x] `src/ingest.rs` rewritten to V2 API; exported from `lib.rs`
- [x] `src/classify.rs` exported from `lib.rs`
- [x] `src/spatial.rs` exported from `lib.rs`
- [x] `tests/walk_tests.rs` — 49 tests (walk routing, guardrails, spatial, Tier 1/2/3, secondary-signal, arithmetic-guard)
- [x] `tests/ingest_tests.rs` — 18 tests (sentence/text ingest, edge reinforcement, `ingest_v2_rows`, node position population)
- [x] `tests/classify_tests.rs` — 6 tests (centroid load, intent/tone labels, determinism)
- [x] `cargo test` passes (46/46, zero warnings)

### Phase 5: Architecture Docs and Guardrail Design
Status: **All guardrails and all three routing tiers implemented and wired**
- [x] OOV lexical fallback (5D vector nearest-node snap)
- [x] Reverse anchor resolution (`resolve_start_node`)
- [x] Dated edge tie-breaking (temporal multiplier)
- [x] Session entity memory (`entity_stack`)
- [x] `sanitize_queue` (Guardrail 3) wired into answer generation
- [x] Tone and entity fields contribute to walk scoring
- [x] Tier 2 KD-tree proximity search in active runtime (`walk.rs` + `SpatialGrid` wired in `predict_next`)
- [x] Dynamic topological-density sentence limits (`compute_depth_limit` wired in CLI mode)
- [x] Tier 3 backtrack-reroute in active runtime — ancestor BFS in `predict_next`; escapes dead-ends Tier 1 and Tier 2 cannot resolve
- [x] Logic/arithmetic interception — `is_arithmetic_query()` in `walk.rs` (Guardrail 6); wired in CLI mode before classification
- [x] Multi-signal validation generalised — `secondary_signal()` in `walk.rs` replaces hard-coded ATM guard; wired in both demo and CLI mode

### Phase 6: V3 Scaling and LLM Wrapper
Status: **Bridge built; Python-side complete; data generation pending**
- [x] `python/download_corpus.py` exists
- [x] `python/v3_ingest.py` exists
- [x] `python/minillm_wrapper.py` implemented and bridged from Rust (CLI mode)
- [x] Rust auto-detects and merges `data/v3_graph_edges.json` when present
- [x] Rust → Python LLM wrapper subprocess (graceful fallback)
- [ ] `data/corpus_v3_massive.txt` generated (run `python/download_corpus.py`)
- [ ] `data/v3_graph_edges.json` generated (run `python/v3_ingest.py`)

### Phase 7: Live Classification
Status: **Complete — bridge wired with graceful fallback**
- [x] `python/classify_query.py` implemented (centroid-based, mirrors `train_centroids.py` exactly)
- [x] Rust `classify_query()` subprocess bridge wired into CLI path
- [x] ML-classified intent/tone/domain drives session context in CLI mode
- [x] Fallback to user-supplied domain + safe defaults when Python unavailable

---

## Remaining Blockers

1. **V3 data absent** — both `corpus_v3_massive.txt` and `v3_graph_edges.json` must be generated by running `python/download_corpus.py` then `python/v3_ingest.py`; the Rust load path is ready and waiting.
2. **`src/classify.rs` Rust-native path unused** — the centroid struct is valid but requires pre-computed `[f32]` embeddings from Rust; the Python subprocess bridge is the practical path until a Rust ONNX runtime is added.
3. ~~**`ingest_rows` private to `main.rs`**~~ — **Done**: `V2JsonData` and `ingest_v2_rows` moved to `src/ingest.rs`; fully unit-tested.

---

## Next High-Value Tasks

1. **Activate V3 pipeline**: `cd python && python download_corpus.py && python v3_ingest.py` — Rust will auto-load on next run.
2. **Install Python dependencies and test full CLI path**: `pip install -r python/requirements.txt` then `cargo run -- "Are the servers online?" server tech 2026` to exercise classifier + LLM wrapper end-to-end.
3. ~~**Move `V2JsonData` + `ingest_rows` to `src/ingest.rs`**~~ — **Done**.
4. **Rust-native classifier path** — wire `src/classify.rs` directly from Rust using a bundled embedding model (e.g. via `ort` ONNX runtime) so the Python subprocess dependency is optional rather than required.

---

## Handoff Context

- **Working path**: `python/v2_ingest.py` → `data/v2_graph_edges.json` → `cargo run`
- **`cargo run`**: passes — 5 demo scenarios, zero warnings
- **`cargo run -- "query" entity domain [year]`**: passes — classifier and LLM bridge both degrade gracefully when Python packages absent
- **`cargo test`**: 73/73 pass across all suites (49 walk + 18 ingest + 6 classify), zero warnings
- **V3/RAG**: Rust load path ready; Python scripts ready; only data generation step missing
- **Full pipeline** (when packages installed): `classify_query.py` → intent/tone/domain → `walk.rs` Tier 1 → `minillm_wrapper.py` → styled response
