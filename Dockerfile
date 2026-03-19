# ─── Stage 1: Rust build ──────────────────────────────────────────────────────
FROM rust:latest AS builder

WORKDIR /build

# Cache dependencies before copying source
COPY Cargo.toml Cargo.lock* ./
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn dummy() {}" > src/lib.rs && \
    cargo build --release 2>/dev/null || true

# Build real source
COPY src ./src
RUN touch src/main.rs src/lib.rs && cargo build --release

# ─── Stage 2: Python runtime ──────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System libraries needed by scikit-learn / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled Rust binary
COPY --from=builder /build/target/release/spse_predictive ./spse_predictive

# Python dependencies
COPY python/requirements.txt ./python/requirements.txt
RUN pip install --no-cache-dir -r python/requirements.txt

# Download spaCy model (correct method: python3 -m spacy download)
RUN python3 -m spacy download en_core_web_sm

# Download NLTK data required by classify_query.py, ingest.py, server.py
RUN python3 -c "\
import nltk; \
nltk.download('punkt_tab',                    quiet=True); \
nltk.download('averaged_perceptron_tagger_eng', quiet=True); \
nltk.download('maxent_ne_chunker_tab',        quiet=True); \
nltk.download('words',                        quiet=True); \
nltk.download('punkt',                        quiet=True); \
"

# Python pipeline scripts and web UI
COPY python/ ./python/
COPY webui/   ./webui/

# data/ is volume-mounted at runtime (corpus files + graph.db + centroids.json)
# We create the directory here so the mount point exists
RUN mkdir -p data

# Entrypoint
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x docker-entrypoint.sh

EXPOSE 5000

ENV SPSE_BINARY=/app/spse_predictive
ENV PORT=5000

ENTRYPOINT ["./docker-entrypoint.sh"]
