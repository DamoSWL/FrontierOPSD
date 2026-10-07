"""Single-device model backend for the frontier algorithm.

Uses the same actor/no-grad teacher idea as SDAR's _compute_teacher_log_probs,
but snapshots the model and caches sampled-token targets only for the current round.
An encoder callback preserves the model's own prompt/action conventions.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .losses import distillation_loss
from .trainer import History, Target


@dataclass(frozen=True)
class TensorTarget:
    student_inputs: dict[str, torch.Tensor]
    tokens: torch.Tensor
    teacher_log_probs: torch.Tensor
    old_log_probs: torch.Tensor


class TorchPolicy:
    """Backend for HF-style causal/VL models with generate() and logits.

    encode(history, hint) returns a single-example processor input dictionary.
    When hint=None it MUST encode only the task and observed environment history.
    decode_action(token_ids) converts generated tokens to an executable action.
    Model parameters must be on one device; distributed FSDP is not supported.
    """

    def __init__(self, model, optimizer, encode: Callable, decode_action: Callable,
                 objective="sampled_opsd", update_steps=1, max_grad_norm=1.0,
                 generation_kwargs=None):
        if objective != "sampled_opsd":
            raise ValueError("Only sampled_opsd is supported; KL is the sole training loss")
        if type(update_steps) is not int or update_steps < 1:
            raise ValueError("update_steps must be a positive integer")
        if not 0 < max_grad_norm < float("inf"):
            raise ValueError("max_grad_norm must be finite and positive")
        self.model, self.optimizer = model, optimizer
        self.encode, self.decode_action = encode, decode_action
        self.objective, self.update_steps = objective, update_steps
        self.max_grad_norm = max_grad_norm
        self._rollout_tokens = {}
        self.generation_kwargs = {"max_new_tokens": 128, "do_sample": True, "temperature": 1.0,
                                  "top_p": 1.0, "top_k": 0, "num_beams": 1}
        self.generation_kwargs.update(generation_kwargs or {})
        if (self.generation_kwargs.get("do_sample") is not True
                or self.generation_kwargs.get("temperature") != 1.0
                or self.generation_kwargs.get("top_p") != 1.0
                or self.generation_kwargs.get("top_k") != 0
                or self.generation_kwargs.get("num_beams") != 1):
            raise ValueError("Sampled OPSD requires unwarped unit-temperature policy sampling")
        if self.generation_kwargs.get("num_return_sequences", 1) != 1:
            raise ValueError("On-policy actions require num_return_sequences=1")

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _inputs(self, history, hint=None):
        inputs = self.encode(deepcopy(history), hint)
        if not isinstance(inputs, dict) or not all(isinstance(value, torch.Tensor) for value in inputs.values()):
            raise TypeError("Encoder must return a dictionary of processor tensors")
        if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
            raise ValueError("Encoder must return one nonempty prompt at a time")
        if not inputs["input_ids"].shape[1]:
            raise ValueError("Empty prompt")
        if "position_ids" in inputs or "labels" in inputs:
            raise ValueError("Let the model compute position_ids; encoder must not return labels")
        return {key: value.detach().to(self.device) for key, value in inputs.items()}

    @torch.no_grad()
    def _generate(self, inputs):
        self.model.eval()
        generated = self.model.generate(**inputs, **self.generation_kwargs)
        if hasattr(generated, "sequences"):
            generated = generated.sequences
        tokens = generated[0, inputs["input_ids"].shape[-1]:].detach().clone()
        if not tokens.numel():
            raise ValueError("Teacher/student generated no action tokens")
        return tokens

    @staticmethod
    def _history_key(history):
        return repr((history.task, history.observations, history.actions))

    def act(self, history: History) -> Any:
        tokens = self._generate(self._inputs(history)).cpu()
        self._rollout_tokens[self._history_key(history)] = tokens.clone()
        return self.decode_action(tokens)

    def begin_rollout(self):
        self._rollout_tokens.clear()

    def freeze(self):
        frozen_model = deepcopy(self.model).eval()
        frozen_model.requires_grad_(False)
        frozen = FrozenTorchPolicy(frozen_model, self.encode, self.decode_action,
                                   self.objective, self.generation_kwargs)
        frozen._rollout_tokens = deepcopy(self._rollout_tokens)
        return frozen

    @staticmethod
    def _logits(model, inputs, tokens):
        inputs = dict(inputs)
        inputs["input_ids"] = torch.cat((inputs["input_ids"], tokens.unsqueeze(0)), dim=-1)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat((inputs["attention_mask"],
                torch.ones_like(tokens).unsqueeze(0)), dim=-1)
        # No KV cache while computing all action-token conditionals.
        logits = model(**inputs, use_cache=False).logits
        return logits[0, -tokens.numel() - 1:-1, :]

    def _loss(self, target):
        payload = target.payload
        inputs = {key: value.to(self.device) for key, value in payload.student_inputs.items()}
        tokens = payload.tokens.to(self.device)
        logits = self._logits(self.model, inputs, tokens)
        mask = torch.ones_like(tokens, dtype=torch.float32)
        log_p = F.log_softmax(logits.float(), -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        return distillation_loss(log_p, payload.teacher_log_probs.to(self.device),
                                 payload.old_log_probs.to(self.device), mask)

    def update(self, target):
        # eval disables dropout but retains autograd, aligning p and cached q.
        self.model.eval()
        for _ in range(self.update_steps):
            self.optimizer.zero_grad(set_to_none=True)
            new_loss = self._loss(target)
            if not torch.isfinite(new_loss):
                raise FloatingPointError("Nonfinite frontier loss")
            new_loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm,
                                                 error_if_nonfinite=True)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return {"frontier/loss": new_loss.detach().item(), "frontier/grad_norm": float(norm)}


class FrozenTorchPolicy(TorchPolicy):
    def __init__(self, model, encode, decode_action, objective, generation_kwargs):
        super().__init__(model, None, encode, decode_action, objective=objective,
                         generation_kwargs=generation_kwargs)

    @torch.no_grad()
    def target(self, history, action, hint, version):
        tokens = self._rollout_tokens[self._history_key(history)].to(self.device)
        if self.decode_action(tokens.cpu()) != action:
            raise ValueError("Original action does not match cached rollout tokens")
        teacher_inputs = self._inputs(history, hint)
        student_inputs = self._inputs(history)
        def score(inputs):
            return F.log_softmax(self._logits(self.model, inputs, tokens).float(), -1).gather(
                -1, tokens.unsqueeze(-1)).squeeze(-1).cpu()
        teacher_log_probs = score(teacher_inputs)
        old_log_probs = score(student_inputs)
        return Target(history, action, TensorTarget(
            {key: value.cpu().clone() for key, value in student_inputs.items()},
            tokens.cpu().clone(), teacher_log_probs, old_log_probs), version)

    def update(self, *args, **kwargs):
        raise RuntimeError("Frozen teacher cannot be updated")
