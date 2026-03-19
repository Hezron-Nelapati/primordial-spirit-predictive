"""
sentence_queue.py  —  Rolling-window sentence embedding buffer for multi-turn context.

Each session is identified by a string session_id (e.g. process PID from Rust).
The queue state is persisted to /tmp/spse_queue_<session_id>.json so it survives
across subprocess calls (Rust spawns a fresh Python process per query).

Usage:
    from sentence_queue import SentenceQueue

    q = SentenceQueue(session_id="12345", window=3)
    q.push(embedding)          # np.ndarray or list[float]
    blended = q.blended()      # weighted average of window, most recent highest weight
    q.save()                   # persist to disk
"""

import json
import os
import numpy as np


class SentenceQueue:
    """Rolling-window embedding buffer with weighted blending.

    Window weights are linearly increasing so the most recent embedding
    contributes most to the blended context vector.

    Args:
        session_id: Unique string identifier for this session (e.g. process PID).
        window:     Maximum number of embeddings to retain (default 3).
    """

    def __init__(self, session_id: str, window: int = 3):
        self.session_id = session_id
        self.window = window
        self._path = f"/tmp/spse_queue_{session_id}.json"
        self._embeddings: list[list[float]] = []
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self):
        """Load queue state from disk if it exists."""
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    data = json.load(f)
                self._embeddings = data.get("embeddings", [])
            except (json.JSONDecodeError, OSError):
                self._embeddings = []

    def save(self):
        """Persist current queue state to disk."""
        try:
            with open(self._path, "w") as f:
                json.dump({"embeddings": self._embeddings}, f)
        except OSError:
            pass  # Non-fatal — single-turn mode unaffected

    # ── queue operations ──────────────────────────────────────────────────────

    def push(self, embedding):
        """Append a new embedding and trim to window size."""
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        self._embeddings.append(embedding)
        if len(self._embeddings) > self.window:
            self._embeddings = self._embeddings[-self.window:]

    def blended(self) -> list[float]:
        """Return a weighted average of all queued embeddings.

        Weights increase linearly from oldest to newest so recent context
        dominates.  Falls back to the single stored embedding when queue
        length is 1, or to the raw input list when queue is empty.
        """
        if not self._embeddings:
            return []
        if len(self._embeddings) == 1:
            return self._embeddings[0]

        n = len(self._embeddings)
        # Linear ramp: [1, 2, ..., n] normalised to sum to 1
        raw_weights = [float(i + 1) for i in range(n)]
        total = sum(raw_weights)
        weights = [w / total for w in raw_weights]

        arrs = [np.array(e) for e in self._embeddings]
        blended = sum(w * a for w, a in zip(weights, arrs))
        return blended.tolist()

    def clear(self):
        """Reset the queue and remove the disk file."""
        self._embeddings = []
        try:
            os.remove(self._path)
        except OSError:
            pass
