"""Separate-model textual frontier diagnosis; no actor calls or action targets."""

import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from frontier_opsd.adapters import CRITIC_INSTRUCTION

logger = logging.getLogger(__name__)


def validate_diagnosis(result, episode):
    if result is None:
        return None
    if not isinstance(result, dict) or set(result) != {"frontier", "hint"}:
        raise ValueError("Critic must return null or exactly {frontier, hint}")
    index, hint = result["frontier"], result["hint"]
    if type(index) is not int or not 0 <= index < len(episode["steps"]):
        raise ValueError("Critic returned an invalid zero-based frontier")
    if not isinstance(hint, str) or not hint.strip():
        raise ValueError("Critic returned an empty hint")
    return result


class ExternalFrontierCritic:
    """Use an independently served model through chat completions or Responses.

    Invalid or truncated diagnoses are skipped. Endpoint failures stop training
    rather than silently substituting the student or skipping every example.
    Credentials are read only from an environment variable, never Hydra config.
    """

    def __init__(self, config):
        settings = config.algorithm.frontier
        self.model = settings.critic_model
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Set +algorithm.frontier.critic_model to the separately served critic model")
        self.api = settings.critic_api
        if self.api not in ("chat_completions", "responses"):
            raise ValueError("Unsupported critic API")
        endpoint = "/responses" if self.api == "responses" else "/chat/completions"
        self.url = settings.critic_base_url.rstrip("/") + endpoint
        self.timeout = settings.critic_timeout
        self.max_tokens = settings.critic_max_tokens
        self.api_key_env = settings.critic_api_key_env

    def __call__(self, episode):
        if episode["success"]:
            return None
        trace = [{"step": index, "before": row["frontier_before"],
                  "action": row["frontier_action"], "after": row["frontier_after"]}
                 for index, row in enumerate(episode["steps"])]
        payload = {"model": self.model, "temperature": 0,
                   "max_tokens": self.max_tokens,
                   "messages": [{"role": "system", "content": CRITIC_INSTRUCTION},
                                {"role": "user", "content": json.dumps({
                                    "success": False, "steps": trace}, ensure_ascii=False)}]}
        if self.api == "responses":
            payload = {"model": self.model, "instructions": CRITIC_INSTRUCTION,
                       "input": payload["messages"][1]["content"],
                       "max_output_tokens": self.max_tokens, "store": False}
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env)
        if self.api == "responses" and not key:
            raise ValueError(f"Set {self.api_key_env} for the hosted critic")
        if key:
            headers["Authorization"] = "Bearer " + key
        request = Request(self.url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"External critic HTTP {error.code}; check model, endpoint, credentials and context limit") from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError("External critic endpoint unavailable; check its server and critic_timeout") from None
        except (ValueError, UnicodeError):
            raise RuntimeError("External critic endpoint returned invalid JSON") from None
        try:
            if self.api == "responses":
                if result.get("status") != "completed":
                    logger.warning("Skipping incomplete external critic diagnosis")
                    return None
                text = "".join(part["text"] for item in result["output"]
                               if item.get("type") == "message"
                               for part in item.get("content", [])
                               if part.get("type") == "output_text").strip()
            else:
                choice = result["choices"][0]
                if choice.get("finish_reason") != "stop":
                    logger.warning("Skipping unfinished external critic diagnosis")
                    return None
                text = choice["message"]["content"].strip()
            if text.startswith("```") and text.endswith("```"):
                text = "\n".join(text.splitlines()[1:-1])
            return validate_diagnosis(json.loads(text), episode)
        except (KeyError, IndexError, AttributeError, ValueError, TypeError):
            logger.warning("Skipping invalid external critic diagnosis")
            return None


def critic_messages(episode):
    trace = [{"step": index, "before": row["frontier_before"],
              "action": row["frontier_action"], "after": row["frontier_after"]}
             for index, row in enumerate(episode["steps"])]
    return [{"role": "system", "content": CRITIC_INSTRUCTION},
            {"role": "user", "content": json.dumps({"success": False, "steps": trace}, ensure_ascii=False)}]


class VLLMFrontierCritic:
    """Offline text critic: load once and call vLLM's Python generate API."""

    def __init__(self, config):
        from vllm import LLM, SamplingParams
        settings = config.algorithm.frontier
        self.enable_thinking = settings.critic_enable_thinking
        self.llm = LLM(model=settings.critic_model,
                       tensor_parallel_size=settings.critic_tensor_parallel_size,
                       distributed_executor_backend="mp",
                       gpu_memory_utilization=settings.critic_gpu_memory_utilization,
                       max_model_len=settings.critic_max_model_len,
                       trust_remote_code=settings.critic_trust_remote_code)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(temperature=0, max_tokens=settings.critic_max_tokens)

    def __call__(self, episode):
        if episode["success"]:
            return None
        tokens = self.tokenizer.apply_chat_template(
            critic_messages(episode), tokenize=True, add_generation_prompt=True,
            enable_thinking=self.enable_thinking)
        if len(tokens) + self.sampling.max_tokens > self.llm.llm_engine.model_config.max_model_len:
            raise ValueError("Critic trace exceeds critic_max_model_len; increase the context limit")
        output = self.llm.generate([{"prompt_token_ids": tokens}], self.sampling, use_tqdm=False)[0].outputs[0]
        if output.finish_reason != "stop":
            logger.warning("Skipping unfinished vLLM critic diagnosis")
            return None
        text = output.text.strip()
        # Qwen3 thinking prompts can already end with <think>, so the output
        # may contain only the closing tag. Discard reasoning in either form.
        if "</think>" in text:
            text = text.partition("</think>")[2].strip()
        elif self.enable_thinking or text.startswith("<think>"):
            logger.warning("Skipping unfinished vLLM critic reasoning")
            return None
        try:
            if text.startswith("```") and text.endswith("```"):
                text = "\n".join(text.splitlines()[1:-1])
            return validate_diagnosis(json.loads(text), episode)
        except (ValueError, TypeError):
            logger.warning("Skipping invalid vLLM critic diagnosis")
            return None

    def diagnose(self, episode):
        return self(episode)


class RayVLLMFrontierCritic:
    """Reserve critic GPUs separately from the student's Ray worker pool."""

    def __init__(self, config):
        import ray
        self.ray = ray
        count = config.algorithm.frontier.critic_tensor_parallel_size
        student_count = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if ray.cluster_resources().get("GPU", 0) < student_count + count:
            raise ValueError(f"vLLM critic needs {count} additional GPUs beyond the {student_count} student GPUs")
        actor = ray.remote(num_cpus=1, num_gpus=count)(VLLMFrontierCritic)
        self.actor = actor.remote(config)
        # Surface model initialization failures before the first diagnosis.
        ray.get(self.actor.diagnose.remote({"success": True}))

    def __call__(self, episode):
        if episode["success"]:
            return None
        # Do not transfer rollout tensors to the critic process.
        trace = {"success": False, "steps": [
            {key: row[key] for key in ("frontier_before", "frontier_action", "frontier_after")}
            for row in episode["steps"]]}
        return self.ray.get(self.actor.diagnose.remote(trace))
