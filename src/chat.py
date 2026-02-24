"""
Mistral-7B NPU Chat - Intel AI Boost Accelerated
A conversational AI powered by OpenVINO on Intel NPU
"""

import openvino_genai as ov_genai
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

try:
    from ddgs import DDGS
    HAS_WEBSEARCH = True
except ImportError:
    HAS_WEBSEARCH = False

# Load environment variables
load_dotenv()

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"

# Configuration from environment
MODEL_NAME = os.getenv("MODEL_NAME", "mistral_npu_cw")
DEVICE = os.getenv("DEVICE", "NPU")
MAX_PROMPT_LEN = int(os.getenv("MAX_PROMPT_LEN", "4096"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful and accurate assistant. Avoid contradictions and obvious errors. If uncertain, say so. Respond in English unless explicitly asked otherwise. Be concise."
)


def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Print welcome banner"""
    print("=" * 50)
    print("  Mistral-7B NPU Chat")
    print("  Intel AI Boost Accelerated")
    print("=" * 50)


def web_search(query, max_results=5):
    """Search the web via DuckDuckGo and return formatted context string."""
    if not HAS_WEBSEARCH:
        return None, "[!] ddgs not installed. Run: pip install ddgs"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return None, "[!] No results found."
        lines = [f"Web search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.get('title', '')}")
            lines.append(f"    {r.get('body', '')}")
            lines.append(f"    Source: {r.get('href', '')}\n")
        return "\n".join(lines), None
    except Exception as e:
        return None, f"[!] Search failed: {e}"


def find_model_path():
    """Find model path - check models/ first, then project root"""
    # First check in models/ directory
    model_in_models = MODEL_DIR / MODEL_NAME
    if model_in_models.exists():
        return str(model_in_models)

    # Fallback to project root (for backward compatibility)
    model_in_root = PROJECT_ROOT / MODEL_NAME
    if model_in_root.exists():
        return str(model_in_root)

    return None


def main():
    clear_screen()
    print_banner()
    print("\nLoading model on {}...\n".format(DEVICE))

    model_path = find_model_path()

    # Check if model exists
    if model_path is None:
        print(f"[!] Model '{MODEL_NAME}' not found!")
        print(f"Searched in: {MODEL_DIR}")
        print(f"             {PROJECT_ROOT}")
        print("\nPlease run 'python src/download.py' first to download the model.")
        sys.exit(1)

    try:
        # Initialize pipeline with compilation cache for faster subsequent loads
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()

        if DEVICE == "NPU":
            pipe = ov_genai.LLMPipeline(
                model_path, DEVICE,
                MAX_PROMPT_LEN=MAX_PROMPT_LEN,
                CACHE_DIR=str(CACHE_DIR),
            )
        else:
            pipe = ov_genai.LLMPipeline(
                model_path, DEVICE,
                CACHE_DIR=str(CACHE_DIR),
            )

        load_time = time.perf_counter() - t0
        pipe.start_chat(SYSTEM_PROMPT)
    except Exception as e:
        print(f"[!] Failed to load model: {e}")
        print(f"Hint: Try setting DEVICE=GPU or DEVICE=CPU in .env file.")
        sys.exit(1)

    print(f"[OK] {DEVICE} ready! (loaded in {load_time:.1f}s)\n")
    web_status = "enabled" if HAS_WEBSEARCH else "disabled (pip install duckduckgo-search)"
    print(f"Commands: /exit  /clear  /reset  /web <query> (websearch: {web_status})\n")
    print("-" * 50)

    # Generation config
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = MAX_NEW_TOKENS
    config.do_sample = True
    config.temperature = TEMPERATURE
    config.top_p = 0.9
    config.repetition_penalty = 1.1

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nBye!")
            break

        if not user_input:
            continue

        # Command handling
        if user_input.lower() in ['/exit', 'quit', 'exit']:
            print("\nBye!")
            pipe.finish_chat()
            break

        if user_input.lower() == '/clear':
            clear_screen()
            continue

        if user_input.lower() == '/reset':
            pipe.finish_chat()
            pipe.start_chat(SYSTEM_PROMPT)
            print("\n[!] Memory cleared.")
            continue

        # /web <query> — search the web and inject results as context
        if user_input.lower().startswith('/web '):
            query = user_input[5:].strip()
            if not query:
                print("\n[!] Usage: /web <your search query>")
                continue
            print(f"\n[Web] Searching: {query} ...")
            context, err = web_search(query)
            if err:
                print(err)
                continue
            print(context)
            print("[Web] Asking the model...\n")
            prompt = (
                f"{context}\n\n"
                f"Based on the web search results above, answer the following question: {query}"
                f" (Reply in English.)"
            )
        else:
            prompt = user_input + " (Reply in English.)"

        print("\nAI > ", end="", flush=True)

        try:
            pipe.generate(
                prompt,
                config,
                streamer=lambda x: print(x, end="", flush=True)
            )
            print()
        except Exception as e:
            if "MAX_PROMPT_LEN" in str(e) or "prompt" in str(e).lower():
                print("\n\n[!] Memory too long, resetting conversation...")
                pipe.finish_chat()
                pipe.start_chat(SYSTEM_PROMPT)
                print("[OK] Conversation reset. Please ask again.")
            else:
                print(f"\n[!] Error: {e}")


if __name__ == "__main__":
    main()
