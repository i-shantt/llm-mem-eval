"""Answer-generation backends.

Two interchangeable options so the end-to-end arm is not blocked on a
particular install:

  TransformersBackend -- local HuggingFace model, works anywhere torch does
  OllamaBackend       -- HTTP to a local ollama server, easier for larger models

Both report prompt and completion token counts so the read-path ledger stays
honest about what the answering model actually consumed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int


class TransformersBackend:
    """Local HF model. Defaults to a small instruct model that fits 6GB."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 64,
        dtype: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._dtype = dtype
        self._model = None
        self._tok = None

    @property
    def name(self) -> str:
        return f"hf:{self.model_name}"

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = self._device
        if dev == "auto":
            dev = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        dtype = torch.float16 if dev in ("cuda", "mps") else torch.float32
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        # AutoModelForCausalLM, not AutoModel -- we need the LM head to generate.
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=dtype
        ).to(dev)
        self._model.eval()
        self._dev = dev

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> Generation:
        import torch

        self._load()
        msgs = [{"role": "user", "content": prompt}]
        text = self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(text, return_tensors="pt").to(self._dev)
        n_prompt = int(inputs.input_ids.shape[1])
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
            )
        gen_ids = out[0][n_prompt:]
        return Generation(
            text=self._tok.decode(gen_ids, skip_special_tokens=True).strip(),
            prompt_tokens=n_prompt,
            completion_tokens=int(gen_ids.shape[0]),
        )


class OllamaBackend:
    """HTTP to a local ollama server (default http://localhost:11434)."""

    def __init__(
        self,
        model_name: str = "qwen2.5:7b-instruct",
        host: str = "http://localhost:11434",
        max_new_tokens: int = 64,
        timeout: int = 300,
    ) -> None:
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"ollama:{self.model_name}"

    def available(self) -> bool:
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5):
                return True
        except Exception:
            return False

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> Generation:
        import json
        import urllib.request

        payload = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens or self.max_new_tokens,
                "temperature": 0.0,
            },
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())
        return Generation(
            text=body.get("response", "").strip(),
            prompt_tokens=int(body.get("prompt_eval_count", 0)),
            completion_tokens=int(body.get("eval_count", 0)),
        )


def build_backend(spec: str, **kw):
    """spec is 'hf:<model>' or 'ollama:<model>'."""
    if spec.startswith("ollama:"):
        return OllamaBackend(model_name=spec.split(":", 1)[1], **kw)
    if spec.startswith("hf:"):
        return TransformersBackend(model_name=spec.split(":", 1)[1], **kw)
    raise ValueError(f"backend spec must start with 'hf:' or 'ollama:', got {spec}")
