# ─── Stage 1: Rust build ──────────────────────────────────────────────────────
FROM rust:latest AS builder

WORKDIR /build

# Cache dependency compilation separately from source
COPY Cargo.toml Cargo.lock* ./
# Create a dummy lib/main so `cargo build` fetches + compiles deps
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn dummy() {}" > src/lib.rs && \
    cargo build --release 2>/dev/null || true

# Now copy real source and build
COPY src ./src
# Touch to force Rust to recompile (dummy files above changed timestamps)
RUN touch src/main.rs src/lib.rs && cargo build --release

# ─── Stage 2: Python runtime ──────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps needed by some Python wheels (e.g. scikit-learn, numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled Rust binary from builder stage
COPY --from=builder /build/target/release/spse_predictive ./spse_predictive

# Python dependencies
COPY python/requirements.txt ./python/requirements.txt
RUN pip install --no-cache-dir -r python/requirements.txt

# Download spaCy model (fixes `python spacy download` — correct method is the module flag)
RUN python3 -m spacy download en_core_web_sm

# Copy Python scripts and web UI
COPY python/ ./python/
COPY webui/  ./webui/

# Data directory — graph.db and corpus files are volume-mounted at runtime
# so we only create the directory here as a mount point.
RUN mkdir -p data

# Expose Flask port
EXPOSE 5000

# Entrypoint triggers first-run DB ingest (warm-up query), then starts Flask
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x docker-entrypoint.sh

ENV SPSE_BINARY=/app/spse_predictive
ENV PORT=5000

ENTRYPOINT ["./docker-entrypoint.sh"]
