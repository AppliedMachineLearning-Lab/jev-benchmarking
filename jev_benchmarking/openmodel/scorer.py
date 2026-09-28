"""Exact next-token scoring with Hugging Face transformers.

For each prompt we need log P(code | prompt) for a handful of single-token answer codes. One forward pass
per prompt gives the full next-token distribution at the last position; we keep only the log-probabilities
of the codes. Prompts are left-padded and packed into batches under a token budget; the output layer is
evaluated at the last position only (`logits_to_keep=1`), which keeps memory flat despite a 262k vocabulary.
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ScoreResult:
    logprobs: list[float]  # log P(code | prompt) under the full vocabulary, one per code
    n_prompt_tokens: int


class HFScorer:
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16", max_batch_tokens: int = 32_768):
        """device="auto" splits the layers over all visible GPUs (for models larger than one GPU)."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if device == "auto":
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), device_map="auto").eval()
            self.device = self.model.get_input_embeddings().weight.device  # inputs go where the embeddings live
        elif device.startswith("cuda"):  # load straight onto the GPU (a 56 GB model would not fit in job RAM first)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), device_map=device).eval()
            self.device = device
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device).eval()
            self.device = device
        self.max_batch_tokens = max_batch_tokens
        pad = self.tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else self.tokenizer.eos_token_id
        self._single = {}

    def is_single_token(self, code: str) -> bool:
        if code not in self._single:
            self._single[code] = len(self.tokenizer.encode(code, add_special_tokens=False)) == 1
        return self._single[code]

    def code_id(self, code: str) -> int:
        ids = self.tokenizer.encode(code, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"answer code {code!r} is not a single token")
        return ids[0]

    def prompt_ids(self, user_text: str) -> list[int]:
        """Chat-formatted prompt ending where the model's answer starts. Thinking is switched off explicitly
        (Gemma 4 defaults to off; Qwen 3.x defaults to on and then emits an empty <think></think> block)."""
        out = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    @torch.inference_mode()
    def score(self, prompts: list[list[int]], codes: list[list[str]]) -> list[ScoreResult]:
        """prompts[i]: token ids; codes[i]: answer codes to score for prompt i."""
        results: list[ScoreResult | None] = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
        batch: list[int] = []
        for i in order:
            longest = len(prompts[batch[0]]) if batch else len(prompts[i])
            if batch and longest * (len(batch) + 1) > self.max_batch_tokens:
                self._run(batch, prompts, codes, results)
                batch = []
            batch.append(i)
        if batch:
            self._run(batch, prompts, codes, results)
        return results

    def _run(self, batch, prompts, codes, results) -> None:
        width = max(len(prompts[i]) for i in batch)
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, i in enumerate(batch):  # left padding: every row ends at the last column
            n = len(prompts[i])
            ids[row, width - n :] = torch.tensor(prompts[i])
            mask[row, width - n :] = 1
        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
        out = self.model(
            input_ids=ids.to(self.device),
            attention_mask=mask.to(self.device),
            position_ids=position_ids.to(self.device),
            logits_to_keep=1,
        )
        logprobs = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1).cpu()
        for row, i in enumerate(batch):
            cid = torch.tensor([self.code_id(c) for c in codes[i]])
            results[i] = ScoreResult(logprobs[row, cid].tolist(), len(prompts[i]))
