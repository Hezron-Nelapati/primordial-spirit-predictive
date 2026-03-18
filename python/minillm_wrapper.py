import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def format_output(graph_fact, user_prompt):
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"  [miniLLM_WRAPPER]: Warming up local conversational formatting engine ({model_id})...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")
    
    messages = [
        {"role": "system", "content": "You are a highly capable conversational AI. You will be provided with a raw factual string retrieved from a mathematical database. Your ONLY job is to rewrite this single fact into a natural, conversational response for the user. Do not add outside knowledge, do not hallucinate, and do not guess. Keep the stylistic response concise and friendly."},
        {"role": "user", "content": f"USER PROMPT: '{user_prompt}'\nRETRIEVED GRAPH FACT: '{graph_fact}'"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("  [miniLLM_WRAPPER]: Generating stylistic fluid response...")
    outputs = pipe(prompt, max_new_tokens=40, do_sample=False, temperature=0.0)
    
    # Extract just the assistant response from the ChatML formatting
    generated_text = outputs[0]["generated_text"].split("<|im_start|>assistant\n")[-1].strip()
    return generated_text

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 minillm_wrapper.py 'Graph Fact' 'User Prompt'")
        sys.exit(1)
        
    fact = sys.argv[1]
    prompt = sys.argv[2]
    
    if "System Fault" in fact:
        print("\n================= 💬 OUTPUT =================\n")
        print(f"🤖 BOT: {fact}")
        print("\n=============================================\n")
    else:
        final_chat = format_output(fact, prompt)
        print("\n================= 💬 OUTPUT =================\n")
        print(f"🤖 BOT: {final_chat}")
        print("\n=============================================\n")
