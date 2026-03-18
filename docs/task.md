# SPSE Predictive Task Status

Last synced against the repository on 2026-03-19.

## Current Architecture Snapshot
- Active Rust runtime is the V2 demo path exported by `src/lib.rs`: `graph`, `reasoning`, and `walk`.
- `src/main.rs` is the only executable entry point. It loads `data/v2_graph_edges.json`, builds a `WordGraph`, seeds `ReasoningModule` state manually, and runs hard-coded demo queries.
- The current walk implementation supports:
  - lexical OOV fallback via `WordNode::compute_lexical_vector`
  - forward edge selection biased by `intent`, `domain`, and optional `dated`
  - reverse anchor resolution via `resolve_start_node`
- The current walk implementation does not yet support:
  - CLI query intake
  - live Python classification
  - KD-tree radial search / Tier 2 routing
  - A* routing / Tier 3 jumps
  - dynamic density-based sentence limits
  - Rust -> `python/minillm_wrapper.py` orchestration
- Active Python scripts:
  - `python/train_centroids.py` writes `data/centroids.json`
  - `python/v2_ingest.py` reads `data/corpus_v2.txt` and writes `data/v2_graph_edges.json`
  - `python/minillm_wrapper.py` is standalone and not yet called from Rust
  - `python/download_corpus.py` targets the Simple Wikipedia `20220301.simple` dataset (`train[:1%]`)
  - `python/v3_ingest.py` exists, but its V3 output files are not present in `data/`
- Legacy code still in-tree:
  - `src/ingest.rs`, `src/spatial.rs`, `src/classify.rs`, and `tests/walk_tests.rs`
  - these files reflect an older V1 API and are not wired through `src/lib.rs`
- Verification snapshot:
  - `cargo run --quiet`: passes
  - `cargo test`: fails because `tests/walk_tests.rs` targets the older V1 API surface

## Phase 1: Repository Foundation
Status: Complete
- [x] `Cargo.toml` and Rust crate scaffolding exist
- [x] `python/requirements.txt` exists
- [x] `data/corpus.txt` and `data/corpus_v2.txt` exist
- [x] `data/centroids.json` is checked in

## Phase 2: Python Training and Ingestion Assets
Status: Complete for local artifacts; not wired into runtime classification
- [x] `python/train_centroids.py` exists and produces `data/centroids.json`
- [x] `python/v2_ingest.py` exists and produces `data/v2_graph_edges.json`
- [x] V2 ingestion uses NLTK tokenization, NLTK NER, regex year extraction, and heuristic intent/tone/domain labeling
- [ ] Centroid inference from `data/centroids.json` is used at runtime
- [ ] spaCy-based NER is used in the active ingestion path
- [ ] `python/requirements.txt` lists all packages used by the checked-in scripts (`nltk`, `datasets`, `transformers`)

## Phase 3: Active Rust V2 Runtime
Status: Complete for the hard-coded demo path
- [x] `src/graph.rs` implements `WordNode`, `WordEdge`, and `WordGraph`
- [x] `src/reasoning.rs` implements session stacks and `sanitize_queue`
- [x] `src/walk.rs` implements OOV lexical fallback, metadata-biased next-token selection, and reverse start-node resolution
- [x] `src/main.rs` loads `data/v2_graph_edges.json` and runs demo scenarios
- [x] Temporal weighting through `dated` edges is implemented
- [x] Pronoun fallback uses the session entity stack
- [ ] Tone-based routing is used in scoring
- [ ] Entity-based routing is used in scoring
- [ ] Queue sanitization is invoked by the executable flow
- [ ] User queries are accepted from the CLI instead of hard-coded examples

## Phase 4: Legacy V1 Surface
Status: Present but stale
- [x] `src/ingest.rs`, `src/spatial.rs`, `src/classify.rs`, and `tests/walk_tests.rs` are still in the repo
- [ ] These modules are re-exported from `src/lib.rs`
- [ ] These modules compile against the current V2 graph/walk API
- [ ] `cargo test` passes
- Note: this is the main reason the test suite is currently out of sync with the executable architecture.

## Phase 5: Architecture Docs and Guardrail Design
Status: Mixed; the design docs are ahead of the code
- [x] `docs/implementation_plan.md`, `docs/walkthrough.md`, and `docs/architecture_ideas.md` document the intended V2+/RAG design
- [x] OOV lexical fallback is implemented
- [x] Reverse anchor resolution is implemented
- [x] Dated edge tie-breaking is implemented
- [x] Session entity memory is implemented
- [ ] Tier 2 KD-tree proximity search is implemented in the active runtime
- [ ] Tier 3 / A* routing is implemented in the active runtime
- [ ] Dynamic topological-density sentence limits are implemented
- [ ] Logic/arithmetic interception is implemented
- [ ] Multi-signal validation is more than a hard-coded demo guard in `src/main.rs`

## Phase 6: V3 Scaling and LLM Wrapper
Status: Partial
- [x] `python/download_corpus.py` exists
- [x] `python/v3_ingest.py` exists
- [x] `python/minillm_wrapper.py` exists
- [ ] `data/corpus_v3_massive.txt` exists in this repo snapshot
- [ ] `data/v3_graph_edges.json` exists in this repo snapshot
- [ ] Rust loads V3 graph data
- [ ] Rust pipes retrieved facts into `python/minillm_wrapper.py`
- [ ] An end-to-end RAG-style query path exists

## Current Blockers / Reality Check
- The repo currently behaves as a local Rust demo over checked-in V2 JSON, not as a full Python-classified, CLI-driven, Rust-to-LLM pipeline.
- The checked-in docs describe several future-facing capabilities that are not yet wired into the executable code.
- `data/centroids.json` and `src/classify.rs` exist, but the active runtime does not consume them.
- `kiddo`, `rusqlite`, and `rand` are declared in `Cargo.toml`, but the active runtime path exported by `src/lib.rs` does not currently use them.
- Auth, authorization, tenant isolation, billing, audit logs, notifications, and production rollout concerns are not applicable to the current local prototype.

## Next High-Value Tasks
- Update or remove the stale V1 test/module surface so `cargo test` reflects the active V2 architecture.
- Add a real CLI query path to `src/main.rs`.
- Connect runtime query parsing/classification to the Python pipeline or port that logic into Rust.
- Add the Rust -> `python/minillm_wrapper.py` handoff only after the deterministic fact path is exposed as a stable interface.

## Handoff Context
- Current phase status: documentation synchronized to the repo as of 2026-03-19.
- Working path today: `python/v2_ingest.py` -> `data/v2_graph_edges.json` -> `cargo run`.
- Known issues: the test suite targets an older architecture; V3/RAG integration is script-only and not wired into the runtime.
- Recommended next step: decide whether to consolidate on the V2 runtime or revive the V1 modules before adding new integration work.
