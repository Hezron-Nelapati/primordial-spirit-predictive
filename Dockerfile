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

# ─── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.11-slim

# GPU build arg — controls which PyTorch wheel index is used.
# Passed via:  docker compose build --build-arg GPU=cuda121
#
#  Value        Hardware              Notes
#  ----------   -------------------  ----------------------------------------
#  cpu          Any / no GPU         Default — works everywhere
#  cuda121      NVIDIA (CUDA 12.1)   Requires NVIDIA Container Toolkit on host
#  cuda118      NVIDIA (CUDA 11.8)   Older NVIDIA drivers
#  rocm61       AMD (ROCm 6.1)       Linux only; requires ROCm drivers on host
#
# Apple Silicon (MPS): run Python directly on macOS — Docker on Mac runs a
# Linux VM and cannot access the Metal GPU. Native python3 train_pipeline.py
# will automatically pick up MPS via gpu_utils.get_device().
ARG GPU=cpu

WORKDIR /app

# System libraries needed by scikit-learn / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled Rust binary
COPY --from=builder /build/target/release/spse_predictive ./spse_predictive

# Install PyTorch with the correct wheel for the target GPU.
# Done before requirements.txt so pip doesn't overwrite it.
RUN if [ "$GPU" = "cuda121" ]; then \
        pip install --no-cache-dir torch==2.3.0 \
            --index-url https://download.pytorch.org/whl/cu121; \
    elif [ "$GPU" = "cuda118" ]; then \
        pip install --no-cache-dir torch==2.3.0 \
            --index-url https://download.pytorch.org/whl/cu118; \
    elif [ "$GPU" = "rocm61" ]; then \
        pip install --no-cache-dir torch==2.3.0 \
            --index-url https://download.pytorch.org/whl/rocm6.1; \
    else \
        pip install --no-cache-dir torch==2.3.0 \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Remaining Python dependencies (torch already installed above)
COPY python/requirements.txt ./python/requirements.txt
RUN pip install --no-cache-dir -r python/requirements.txt

# Download spaCy model
RUN python3 -m spacy download en_core_web_sm

# Download NLTK data required by classify_query.py, ingest.py, ingest_wiki.py
RUN python3 -c "\
import nltk; \
nltk.download('punkt_tab',                      quiet=True); \
nltk.download('averaged_perceptron_tagger_eng', quiet=True); \
nltk.download('punkt',                          quiet=True); \
"

# Python pipeline scripts and web UI
COPY python/ ./python/
COPY webui/   ./webui/

# data/ is volume-mounted at runtime (corpus files + graph.db + centroids.json)
RUN mkdir -p data

COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x docker-entrypoint.sh

EXPOSE 5000

ENV SPSE_BINARY=/app/spse_predictive
ENV PORT=5000

ENTRYPOINT ["./docker-entrypoint.sh"]
