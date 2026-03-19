"""
minillm_wrapper.py — Stylistic LLM formatter using SmolLM2-135M-Instruct.

Exposes:
  load_model()         — pre-load model (call once at startup)
  style(fact, query)   — module-level API, returns formatted string
  format_output(...)   — legacy alias

CLI:
  python3 minillm_wrapper.py 'graph_fact' 'user_prompt'
"""
import sys
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Module-level singleton — loaded once, reused on every style() call
_pipe = None


def load_model():
    """Pre-load the LLM pipeline. Safe to call multiple times."""
    global _pipe
    if _pipe is None:
        print(f"  [miniLLM_WRAPPER]: Loading {_MODEL_ID} (once)…", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        model     = AutoModelForCausalLM.from_pretrained(_MODEL_ID)
        _pipe     = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")
        print("  [miniLLM_WRAPPER]: Model ready.", file=sys.stderr)
    return _pipe


def style(graph_fact: str, user_prompt: str) -> str:
    """Clean up a raw graph walk result into readable text.

    SmolLM2-135M-Instruct cannot reliably ground its output to an external
    fact — it generates from its own parametric weights instead, producing
    hallucinations like wrong names, wrong dates, and invented narratives.
    This function replaces LLM generation with deterministic Python cleanup
    so the answer always reflects what the graph walk actually found.
    """
    if "System Fault" in graph_fact:
        return graph_fact

    text = graph_fact.strip()

    # Strip leading corpus-boundary tokens.  The graph walker anchors at ``
    # or '' sentence-boundary nodes in the corpus; these appear verbatim at
    # the start of the raw fact when the reverse-walk reaches a sentence head.
    while text and text[0] in "`'\"":
        text = text[1:].lstrip()

    # Capitalise first letter.
    if text:
        text = text[0].upper() + text[1:]

    # Ensure terminal punctuation so the response reads as a complete thought.
    if text and text[-1] not in ".!?":
        text += "."

    return text


# Legacy alias used by existing code
def format_output(graph_fact: str, user_prompt: str) -> str:
    return style(graph_fact, user_prompt)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 minillm_wrapper.py 'Graph Fact' 'User Prompt'")
        sys.exit(1)

    fact   = sys.argv[1]
    prompt = sys.argv[2]
    result = style(fact, prompt)
    print("\n================= 💬 OUTPUT =================\n")
    print(f"🤖 BOT: {result}")
    print("\n=============================================\n")
