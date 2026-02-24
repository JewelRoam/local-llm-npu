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
    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.key_binding import KeyBindings
    HAS_MULTILINE = True
except ImportError:
    HAS_MULTILINE = False

try:
    from ddgs import DDGS
    HAS_WEBSEARCH = True
except ImportError:
    HAS_WEBSEARCH = False

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False


def translate_to_english(text):
    """Translate text to English. Returns (translated, error)."""
    if not HAS_TRANSLATOR:
        return text, None
    try:
        result = GoogleTranslator(source='auto', target='en').translate(text)
        return result, None
    except Exception as e:
        return text, f"[!] Translation failed: {e}"


def read_input(prompt_text):
    """Read user input. Supports multi-line with Alt+Enter if prompt_toolkit is available."""
    if not HAS_MULTILINE:
        return input(prompt_text)

    kb = KeyBindings()

    @kb.add('enter')
    def submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add('escape', 'enter')
    def newline(event):
        event.current_buffer.insert_text('\n')

    return pt_prompt(
        prompt_text,
        multiline=True,
        key_bindings=kb,
        prompt_continuation='     ',
    )

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

LOGIC_AUDIT_INSTRUCTION = """

[LOGIC AUDIT MODE — follow this workflow before answering]

**[Logic Audit]**
- Step A (Grounding): Extract all numeric values, entities, and key claims. Map them to real-world scale (e.g., monetary amounts vs. GDP/market cap, distances, time costs).
- Step B (Conflict Detection): Actively search for flaws in a naive answer:
  - Efficiency contradiction: Does the action cost more than it gains?
  - Physical necessity: Does the task require specific conditions or tools?
  - Source reliability: Is any claim in an AI hallucination-prone zone?
- Summarize what conflicts (if any) were found.

**[Final Verified Answer]**
Provide the corrected, logically consistent answer after completing the audit above."""


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
    multiline_status = "Alt+Enter for newline" if HAS_MULTILINE else "disabled (pip install prompt_toolkit)"
    print(f"Commands: /exit  /clear  /reset  /web <query> (websearch: {web_status})")
    print(f"          /logic — deep logic audit mode  |  multi-line: {multiline_status}\n")
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
            user_input = read_input("\nYou > ").strip()
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

        # Detect /logic mode (can appear anywhere in the message)
        logic_mode = '/logic' in user_input.lower()
        clean_input = user_input.replace('/logic', '').replace('/LOGIC', '').strip()
        if logic_mode:
            print("\n[Logic] Deep audit mode activated.")

        # /web <query> — search the web and inject results as context
        # Match /web followed by space, newline, or end-of-string
        web_body = None
        if clean_input.lower().startswith('/web'):
            rest = clean_input[4:]
            if not rest or rest[0] in (' ', '\n', '\r', '\t'):
                web_body = rest.strip()

        if web_body is not None:
            if not web_body:
                print("\n[!] Usage: /web <query>  or  /web <query> /logic")
                continue
            # Use first non-empty line as search query; full body as the question
            non_empty_lines = [l.strip() for l in web_body.split('\n') if l.strip()]
            search_query = non_empty_lines[0] if non_empty_lines else web_body[:100]

            # Translate query to English for better search results
            eng_query, t_err = translate_to_english(search_query)
            if t_err:
                print(t_err)
            elif eng_query != search_query:
                print(f"\n[Web] Translated query: {eng_query}")
            print(f"\n[Web] Searching: {eng_query} ...")
            context, err = web_search(eng_query)
            if err:
                print(err)
                continue
            print(context)
            print("[Web] Asking the model...\n")
            prompt = (
                f"{context}\n\n"
                f"Based on the web search results above, answer the following question:\n{web_body}"
                f"\n(Reply in English.)"
            )
        else:
            prompt = clean_input + " (Reply in English.)"

        if logic_mode:
            prompt += LOGIC_AUDIT_INSTRUCTION

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
