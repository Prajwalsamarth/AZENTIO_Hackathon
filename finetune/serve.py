"""MLXClient -- serves the fine-tuned model behind OllamaClient's interface.

Mirrors `.available()` and `.chat()` so it drops straight into
slm.adjudicate()'s retry-and-fallback machinery with no changes there.

The important difference from Ollama: MLX has no schema-constrained decoding.
The model must hold JSON format from training alone, so chat() extracts the
first balanced {...} span rather than trusting the output to be pure JSON.
"""

import json
import time

import ft_config

_CACHE = {}


def _load(adapter_path=None):
    key = str(adapter_path)
    if key in _CACHE:
        return _CACHE[key]
    from mlx_lm import load
    model, tokenizer = load(
        ft_config.BASE_MODEL,
        adapter_path=str(adapter_path) if adapter_path else None,
    )
    _CACHE[key] = (model, tokenizer)
    return _CACHE[key]


def extract_json(text: str):
    """First balanced {...} span. Cheap guard against leading prose."""
    depth, start, in_string, escaped = 0, None, False, False
    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
    return None


class MLXClient:
    """Drop-in replacement for slm.OllamaClient, backed by MLX."""

    def __init__(self, adapter_path=None, model=None, max_tokens=None):
        self.adapter_path = adapter_path
        self.model_name = model or ft_config.BASE_MODEL
        self.max_tokens = max_tokens or ft_config.MLX_MAX_TOKENS

    def available(self) -> bool:
        try:
            _load(self.adapter_path)
            return True
        except Exception:
            return False

    def chat(self, messages, schema=None, options=None):
        """schema is accepted for interface parity and intentionally unused --
        MLX cannot constrain decoding. Returns (content, elapsed_seconds)."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = _load(self.adapter_path)
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)

        max_tokens = (options or {}).get("num_predict", self.max_tokens)
        started = time.time()
        text = generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False,
            sampler=make_sampler(temp=ft_config.MLX_TEMPERATURE),
        )
        elapsed = time.time() - started

        span = extract_json(text)
        return (span if span is not None else text), elapsed


def base_client():
    """Untuned baseline, for the before/after comparison."""
    return MLXClient(adapter_path=None)


def tuned_client():
    adapter = ft_config.ADAPTER_DIR
    if not (adapter / "adapters.safetensors").exists():
        raise RuntimeError(f"no trained adapter at {adapter}; run `train` first")
    return MLXClient(adapter_path=adapter)
