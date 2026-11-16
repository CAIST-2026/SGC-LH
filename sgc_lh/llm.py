"""LLM-guided semantic soft weight scoring (eq:prompt, eq:llm_weight).

Two backends:

1.  QwenLLMScorer  -- the paper's configuration: Qwen-7B served with 4-bit
    weight-only quantisation (paper uses vLLM; this implementation uses
    transformers + bitsandbytes, same quantisation principle). The prompt
    P(e_s, e_t) = instruction || name(e_s) || attr(e_s) || name(e_t) || attr(e_t)
    ends at a position whose next-token distribution is restricted to the two
    single-token verbalizers omega_Yes / omega_No, and

        W^LLM = exp(l_Yes) / (exp(l_Yes) + exp(l_No))          (eq:llm_weight)

    W^LLM is a *confidence score*, not a calibrated probability (affirmative
    bias measured in the manuscript's prior audit).

2.  HeuristicLLMScorer -- a deterministic string-similarity scorer that plays
    the same role in the pipeline (same interface, same output range) so the
    full two-stage pipeline can be exercised on CPU machines without a 7B
    model. It is NOT the paper's model and will NOT reproduce the paper's
    numbers; it exists only to validate plumbing and to serve as an offline
    dry-run backend.

Set config.llm_backend = "auto" to pick Qwen when CUDA is available.
"""

from __future__ import annotations

import difflib
from abc import ABC, abstractmethod

import numpy as np
import torch

INSTRUCTION = (
    "Decide whether the following two entity descriptions refer to the same "
    "real-world object. Answer with exactly one word, Yes or No.\n"
)


class BaseLLMScorer(ABC):
    """Batch interface: texts of candidate pairs -> W^LLM in (0, 1)."""

    @abstractmethod
    def score_pairs(self, pair_texts: list[tuple[dict, dict]]) -> np.ndarray:
        ...


class QwenLLMScorer(BaseLLMScorer):
    """Qwen-7B, 4-bit weight-only quantisation, two-token verbalizer."""

    def __init__(self, model_name: str = "Qwen/Qwen-7B", device: str = "cuda",
                 batch_size: int = 16, max_prompt_chars: int = 1500,
                 yes_token: str = "Yes", no_token: str = "No"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        try:  # 4-bit weight-only quantisation (paper: vLLM 4-bit; ~4.5 GB static)
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb, device_map=device, trust_remote_code=True
            )
        except Exception:  # fall back to bf16 if bitsandbytes is unavailable
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
            )
        self.model.eval()
        self.batch_size = batch_size
        self.max_prompt_chars = max_prompt_chars
        self.yes_id = self.tokenizer.encode(yes_token, add_special_tokens=False)
        self.no_id = self.tokenizer.encode(no_token, add_special_tokens=False)
        assert len(self.yes_id) == 1 and len(self.no_id) == 1, (
            "verbalizer tokens must be single tokens (manuscript: pre-verified single-token verbalizers)"
        )
        self.yes_id, self.no_id = self.yes_id[0], self.no_id[0]

    def _prompt(self, a: dict, b: dict) -> str:
        p = (INSTRUCTION
             + "Entity A name: " + str(a.get("name", "")) + "\n"
             + "Entity A attributes: " + str(a.get("attr", ""))[:400] + "\n"
             + "Entity B name: " + str(b.get("name", "")) + "\n"
             + "Entity B attributes: " + str(b.get("attr", ""))[:400] + "\n"
             + "Answer (Yes/No):")
        return p[-self.max_prompt_chars:]

    @torch.no_grad()
    def score_pairs(self, pair_texts: list[tuple[dict, dict]]) -> np.ndarray:
        out = np.empty(len(pair_texts), dtype=np.float32)
        ids_yes, ids_no = [self.yes_id], [self.no_id]
        for s in range(0, len(pair_texts), self.batch_size):
            chunk = pair_texts[s:s + self.batch_size]
            prompts = [self._prompt(a, b) for a, b in chunk]
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                 padding_side="left").to(self.model.device)
            logits = self.model(**enc).logits[:, -1, :]           # next-token distribution
            l_yes = logits[:, ids_yes[0]]
            l_no = logits[:, ids_no[0]]
            # eq:llm_weight -- two-class softmax over the verbalizer logits
            w = torch.softmax(torch.stack([l_yes, l_no], dim=1), dim=1)[:, 0]
            out[s:s + len(chunk)] = w.float().cpu().numpy()
        return out


class HeuristicLLMScorer(BaseLLMScorer):
    """Deterministic stand-in (offline dry-runs only; NOT the paper's model).

    Emulates the affirmative-bias profile qualitatively (score inflated towards
    1 for superficially similar names) so the affine fusion path is exercised.
    """

    def score_pairs(self, pair_texts: list[tuple[dict, dict]]) -> np.ndarray:
        out = np.empty(len(pair_texts), dtype=np.float32)
        for k, (a, b) in enumerate(pair_texts):
            na, nb = str(a.get("name", "")), str(b.get("name", ""))
            aa, ab = str(a.get("attr", ""))[:200], str(b.get("attr", ""))[:200]
            sim_name = difflib.SequenceMatcher(None, na.lower(), nb.lower()).ratio()
            sim_attr = difflib.SequenceMatcher(None, aa.lower(), ab.lower()).ratio()
            raw = 0.7 * sim_name + 0.3 * sim_attr
            # squash into (0,1) with an affirmative-leaning floor, mimicking bias
            out[k] = float(np.clip(0.15 + 0.85 * raw, 1e-6, 1.0 - 1e-6))
        return out


def make_scorer(backend: str = "auto", model_name: str = "Qwen/Qwen-7B",
                device: str = "auto", batch_size: int = 16,
                max_prompt_chars: int = 1500) -> BaseLLMScorer:
    if backend == "heuristic":
        return HeuristicLLMScorer()
    if backend == "qwen":
        dev = "cuda" if device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
        if dev == "cpu":
            raise RuntimeError("backend='qwen' requires CUDA; use backend='heuristic' for CPU dry-runs")
        return QwenLLMScorer(model_name, device=dev, batch_size=batch_size,
                             max_prompt_chars=max_prompt_chars)
    # auto
    if torch.cuda.is_available():
        return QwenLLMScorer(model_name, device="cuda", batch_size=batch_size,
                             max_prompt_chars=max_prompt_chars)
    return HeuristicLLMScorer()
