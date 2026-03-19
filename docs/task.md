# SPSE Predictive Task Status

Last synced against the repository on 2026-03-19. Updated gaps added 2026-03-19.

---

## Current Architecture Snapshot

### Execution Flow — Demo Mode (`cargo run`)
```
data/v2_graph_edges.json  [+ data/v3_graph_edges.json if present]
    ↓ ingest_v2_rows()  (src/ingest.rs)
WordGraph (merged V2 + V3 edges; edge reinforcement on duplicates)
    ↓ SpatialGrid::build()
SpatialGrid (3D KD-tree on lexical-density node positions)
    ↓
ReasoningModule + WalkConfig (compute_depth_limit per entity; reset_session between scenarios)
    ↓
predict_next()  [Tier 1 → Tier 2 KD-tree → Tier 3 backtrack-reroute]
    ↓
5 demo scenario outputs printed to stdout
```

### Execution Flow — CLI Mode (`cargo run -- "query" entity domain [year]`)
```
data/v2_graph_edges.json  [+ data/v3_graph_edges.json if present]
    ↓ ingest_v2_rows()  (src/ingest.rs)
WordGraph + SpatialGrid
    ↓
Guardrail 6: is_arithmetic_query() — abort if arithmetic expression detected
    ↓
python/classify_query.py  →  {intent, tone, domain}   [graceful fallback if Python unavailable]
    ↓
ReasoningModule::sanitize_queue()  [Guardrail 3]
    ↓
secondary_signal() + is_reachable()  [multi-signal reachability guard]
    ↓
predict_next()  [Tier 1 → Tier 2 KD-tree → Tier 3 backtrack-reroute]
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
- `position: [f32; 3]` populated at ingest time from lexical-density coordinates (`len`, `vowels/len`, `unique/len`).
- Status: **Complete and wired**.

### `src/reasoning.rs`
- `SessionalMemory` — four `Vec<String>` stacks: intent, tone, domain, entity.
- `sanitize_queue()` — Guardrail 3, called inside `generate_dynamic_answer()` before every walk.
- `update_context()` — pushes to all four stacks; skips "Pronoun" entity entries.
- `reset_session()` — clears all four stacks; used between independent conversations.
- Status: **Complete and fully wired**.

### `src/walk.rs`
- `predict_next()` — three-tier cascade: Tier 1 multi-signal edge scoring (OOV snap, intent ×2.0, domain ×2.0, tone ×2.0, entity ×1.5, temporal multiplier) → Tier 2 KD-tree radial fallback → Tier 3 ancestor backtrack-reroute.
- `resolve_start_node()` — reverse-walk up to 20 hops to find sentence anchor; domain + temporal biased.
- `is_reachable()` — BFS reachability with `max_hops` bound; used by multi-signal guard.
- `compute_depth_limit()` — 2-hop topology count → dynamic sentence depth (1–4).
- `secondary_signal()` — scans query tokens for a secondary graph-resident entity signal.
- `is_arithmetic_query()` — Guardrail 6; two-condition gate (numeric token + arithmetic signal).
- Status: **All three tiers implemented and wired. All guardrails active.**

### `src/classify.rs`
- `Classifier::load(path)`, `Classifier::intent(emb_full, emb_pos)`, `Classifier::tone(...)`, `Classifier::domain(...)`.
- `CentroidStore` includes domain fields (`domain_labels`, `domain_full_centroids`, `domain_pos_centroids`) with `#[serde(default)]` for backward compatibility with pre-Phase-9 `centroids.json`.
- `domain()` returns `"general"` when domain centroids absent.
- Exported from `lib.rs`; tested in `tests/classify_tests.rs` (9 tests).
- **Not called from `src/main.rs`** — requires pre-computed float embeddings from Python side. The Python classify bridge (`python/classify_query.py`) handles this for CLI queries.
- Status: **Exported; struct current with Phase 9; Rust-side inference unused at runtime; Python-side bridge is the active path.**

### `src/spatial.rs`
- `SpatialGrid` wrapping `kiddo::KdTree<f32, 3>` with `build()`, `query_radius()`, `query_nearest()`.
- Built from `graph.nodes` immediately after corpus load in `main.rs`.
- Passed as `Some(&spatial)` to `predict_next` in both CLI and demo modes — activates Tier 2 routing.
- Status: **Exported, built at runtime, fully integrated into Tier 2.**

### `src/ingest.rs`
- `ingest_text()`, `ingest_sentence()` — plain-text corpus ingestion with edge reinforcement.
- `ingest_v2_rows()` + `V2JsonData` — primary corpus load path for tagged JSON rows; edge reinforcement on identical `(from, to, intent, domain, dated)` tuples.
- `GraphStats::compute()` / `GraphStats::report()` — node count, edge count, avg out-degree.
- Status: **Exported, fully wired as the primary ingest path; unit-tested.**

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

### `python/classify_query.py`
- Runtime classification for CLI queries.
- Accepts `sys.argv[1]` (query string), optional `sys.argv[2]` (centroids path), optional `--session-id ID`.
- Mirrors `train_centroids.py` POS tag sets and blended distance formula exactly.
- **Intent/tone**: `_nearest_blended()` using centroid model.
- **Domain**: `_nearest_blended()` using domain centroids when present; falls back to `_keyword_domain()` for old `centroids.json`.
- **NER** (`_ner_entities()`): spaCy `en_core_web_sm` extracts `PERSON/ORG/GPE/PRODUCT` spans; graceful `[]` fallback if spaCy absent.
- **Session queue**: imports `SentenceQueue` when `--session-id` provided; blended rolling-window embedding used for multi-turn context; graceful fallback if unavailable.
- JSON output: `{"intent": ..., "tone": ..., "domain": ..., "entities": [...]}`.
- All diagnostics to stderr; single JSON line to stdout.
- Rust calls this via `std::process::Command` passing `--session-id <PID>`; falls back to defaults if unavailable.
- Status: **Implemented; requires `sentence-transformers`, `nltk`, `spacy` + `en_core_web_sm` model**.

### `python/minillm_wrapper.py`
- `HuggingFaceTB/SmolLM2-135M-Instruct`, 135 M params, CPU, temperature 0.0.
- Accepts `sys.argv[1]` (graph fact) + `sys.argv[2]` (user prompt).
- Rust calls this via `std::process::Command` in CLI mode; falls back to raw fact if unavailable.
- Status: **Implemented and bridged; requires `transformers` + model cache (~270 MB)**.

### `python/download_corpus.py`
- Downloads Simple English Wikipedia `20220301.simple` → `data/corpus_v3_massive.txt`.
- Status: **Script ready; not executed**.

### `python/sentence_queue.py`
- `SentenceQueue(session_id, window=3)` — rolling window embedding buffer.
- Persists state to `/tmp/spse_queue_<session_id>.json` so it survives across subprocess calls.
- Blended embedding = linearly-weighted average (oldest → weight 1, newest → weight N).
- `push()`, `blended()`, `save()`, `clear()` API.
- Status: **Implemented; imported by `classify_query.py` when `--session-id` is provided**.

### `python/requirements.txt`
```
sentence-transformers
scikit-learn
numpy
nltk
datasets
transformers
spacy
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
| `kiddo = "4"` | ✓ | ✓ — `SpatialGrid` wraps `kiddo::KdTree`; used in Tier 2 routing |
| `rusqlite` | ✗ removed | Was unused; removed |
| `rand` | ✗ removed | Was unused; removed |

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
- [x] spaCy NER in runtime classification path (`classify_query.py` `_ner_entities()`); NLTK used in corpus ingestion (`v2_ingest.py`)
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
- [x] `tests/walk_tests.rs` — 55 tests (walk routing, guardrails, spatial, Tier 1/2/3, secondary-signal, arithmetic-guard, reset_session, WalkMode)
- [x] `tests/ingest_tests.rs` — 18 tests (sentence/text ingest, edge reinforcement, `ingest_v2_rows`, node position population)
- [x] `tests/classify_tests.rs` — 6 tests (centroid load, intent/tone labels, determinism)
- [x] `cargo test` passes (82/82 across all suites, zero warnings)

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

### Phase 8: Intent-Polymorphic Walk Strategies
Status: **Complete**

- [x] `WalkMode` enum (`Forward`, `Explain`, `Question`) added to `walk.rs`; `WalkMode::from_intent()` maps intent strings
- [x] `WalkConfig.mode` field carries mode into every `predict_next` call
- [x] `score_edges_explain()` — selects target with most onward edges (widest topological coverage)
- [x] `score_edges_question()` — selects target closest in BFS hops to active entity anchor; `bfs_distance()` helper
- [x] Dispatch at Tier 1 in `predict_next` based on `config.mode`
- [x] CLI mode: `WalkMode::from_intent(&intent)` used at `WalkConfig` construction
- [x] Demo mode: explicit modes set per scenario (`Question` for Q&A, `Forward` for statements, `Explain` for explanations)
- [x] 3 new tests: explain prefers high out-degree, question routes toward entity anchor, `from_intent` mapping

### Phase 9: Domain Centroid Classification
Status: **Complete**

`classify_query.py` uses a keyword heuristic for domain (same map as `v2_ingest.py`). The `implementation_plan.md` calls for three-tier centroid models covering intent, tone, *and* domain.

- [x] Add domain labels to training data in `train_centroids.py` (tech, finance, science, geography, general — 5 examples each)
- [x] Train and store domain centroids in `data/centroids.json` alongside intent/tone centroids (`domain_labels`, `domain_full_centroids`, `domain_pos_centroids`)
- [x] Update `classify_query.py` to call `_nearest_blended()` for domain; `DOMAIN_TAGS = {"NN","NNS","NNP","NNPS"}` (nouns carry domain signal)
- [x] Backward-compatible: if `domain_labels` absent from store, falls back to `_keyword_domain()` gracefully
- [ ] Retrain `data/centroids.json` (re-run `python/train_centroids.py`) — must be done manually after this change
- [ ] Update `classify_tests.rs` to include domain label validation

### Phase 10: Runtime NER on Live Query Text
Status: **Complete**

Entity extraction currently comes from pre-labelled corpus rows (`row.entities`). Live queries arrive with no entity annotation. The plan calls for NER on the user query itself so `entity_stack` is populated from query content, not just corpus metadata.

- [x] Add `spacy` to `python/requirements.txt`
- [x] `classify_query.py`: `_ner_entities()` — loads `en_core_web_sm`, extracts `PERSON`, `ORG`, `GPE`, `PRODUCT` spans; graceful fallback to `[]` on `ImportError`/`OSError`
- [x] JSON output now includes `"entities": [...]` field alongside intent/tone/domain
- [x] `src/main.rs` CLI path: `classify_query()` returns 4-tuple `(intent, tone, domain, Vec<String>)`; merges NER entities with CLI-supplied entity before calling `reasoning.update_context()`
- [x] `V2JsonData` already has `entities: Vec<String>` — no Rust struct change needed
- [ ] Tests: mock classifier output with entity field, verify entity_stack is populated (future)

### Phase 11: Sentence Queuing (Python Sensory Pipeline)
Status: **Complete**

The `implementation_plan.md` specifies a Sentence Queue to prevent embedding quality degradation in multi-turn conversations. Currently each query is classified in isolation; no buffering occurs.

- [x] Create `python/sentence_queue.py` — `SentenceQueue` class; rolling window of last N (default 3) embeddings; persisted to `/tmp/spse_queue_<session_id>.json`
- [x] Blended embedding = linearly-weighted average of window (oldest → weight 1, newest → weight N, normalised)
- [x] `classify_query.py` imports and uses the queue when `--session-id` arg provided; queue failure falls back gracefully to raw embedding
- [x] Rust passes `--session-id <PID>` to `classify_query.py` subprocess (`std::process::id()`)
- [x] Single-turn (no `--session-id`) continues to work unchanged — queue is fully opt-in

---

## Remaining Blockers

1. **V3 data absent** — both `corpus_v3_massive.txt` and `v3_graph_edges.json` must be generated by running `python/download_corpus.py` then `python/v3_ingest.py`; the Rust load path is ready and waiting.
2. **`src/classify.rs` Rust-native path unused** — the centroid struct is valid but requires pre-computed `[f32]` embeddings from Rust; the Python subprocess bridge is the practical path until a Rust ONNX runtime is added.
3. ~~**Intent-polymorphic walk not implemented**~~ — **Done** (Phase 8): `WalkMode::Forward/Explain/Question` dispatched in `predict_next`.
4. ~~**Domain classification is heuristic**~~ — **Done** (Phase 9): `train_centroids.py` now produces `domain_full_centroids` + `domain_pos_centroids`; `classify_query.py` uses `_nearest_blended()` with keyword fallback. Requires re-running `train_centroids.py` to refresh `centroids.json`.
5. ~~**No runtime NER on live queries**~~ — **Done** (Phase 10): `_ner_entities()` in `classify_query.py` extracts PERSON/ORG/GPE/PRODUCT spans via spaCy; Rust merges them into `entity_stack`.
6. ~~**No sentence queuing**~~ — **Done** (Phase 11): `python/sentence_queue.py` rolling window; `classify_query.py` uses blended embedding when `--session-id` provided; Rust passes PID as session ID.

---

## Next High-Value Tasks

1. ~~**Phase 8 — Intent-polymorphic walk**~~ — Done.
2. ~~**Phase 9 — Domain centroids**~~ — Done (re-run `train_centroids.py` to refresh `centroids.json`).
3. ~~**Phase 10 — Runtime NER**~~ — Done.
4. ~~**Phase 11 — Sentence queuing**~~ — Done.
5. **Activate V3 pipeline**: `cd python && python download_corpus.py && python v3_ingest.py` — Rust will auto-load on next run.

---

## Handoff Context

- **Working path**: `python/v2_ingest.py` → `data/v2_graph_edges.json` → `cargo run`
- **`cargo run`**: passes — 5 demo scenarios, zero warnings
- **`cargo run -- "query" entity domain [year]`**: passes — classifier and LLM bridge both degrade gracefully when Python packages absent
- **`cargo test`**: 82/82 pass across all suites (55 walk + 18 ingest + 9 classify), zero warnings
- **V3/RAG**: Rust load path ready; Python scripts ready; only data generation step missing
- **Full pipeline** (when packages installed): `classify_query.py` → intent/tone/domain → `walk.rs` Tier 1 → `minillm_wrapper.py` → styled response
