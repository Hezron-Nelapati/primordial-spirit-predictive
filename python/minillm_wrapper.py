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
    """Format a raw graph fact into a conversational response.
    Returns the fact unchanged on error or if it contains 'System Fault'."""
    if "System Fault" in graph_fact:
        return graph_fact

    try:
        pipe = load_model()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a highly capable conversational AI. You will be provided "
                    "with a raw factual string retrieved from a mathematical database. "
                    "Your ONLY job is to rewrite this single fact into a natural, "
                    "conversational response for the user. Do not add outside knowledge, "
                    "do not hallucinate, and do not guess. Keep the response concise and friendly."
                ),
            },
            {
                "role": "user",
                "content": f"USER PROMPT: '{user_prompt}'\nRETRIEVED GRAPH FACT: '{graph_fact}'",
            },
        ]
        tokenizer = pipe.tokenizer
        prompt    = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs   = pipe(prompt, max_new_tokens=40, do_sample=False, temperature=0.0)
        return outputs[0]["generated_text"].split("<|im_start|>assistant\n")[-1].strip()
    except Exception as exc:
        print(f"  [miniLLM_WRAPPER]: style() failed ({exc}) — returning raw fact.", file=sys.stderr)
        return graph_fact


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
