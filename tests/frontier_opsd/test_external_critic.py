"""Check critic wire formats and rejection behavior without paid/network calls."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("frontier_external_critic", ROOT / "verl/trainer/frontier/critic.py")
critic_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(critic_module)


class ExternalCriticTests(unittest.TestCase):
    def config(self, api="chat_completions"):
        return SimpleNamespace(algorithm=SimpleNamespace(frontier=SimpleNamespace(
            critic_model="separate-large-model", critic_api=api,
            critic_base_url="http://critic-host:8001/v1/", critic_api_key_env="CRITIC_TEST_KEY",
            critic_timeout=120, critic_max_tokens=512)))

    def episode(self):
        return {"success": False, "steps": [
            {"frontier_before": "Task: find red item", "frontier_action": "select blue item",
             "frontier_after": "Wrong item", "input_ids": object()},
            {"frontier_before": "Wrong item", "frontier_action": "finish", "frontier_after": "Failed"}]}

    def reply(self, text, api="chat_completions", finished=True):
        if api == "responses":
            return {"status": "completed" if finished else "incomplete", "output": [
                {"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": text}]}]}
        return {"choices": [{"finish_reason": "stop" if finished else "length", "message": {"content": text}}]}

    def call_with_reply(self, reply, api="chat_completions"):
        import io
        endpoint = critic_module.ExternalFrontierCritic(self.config(api))
        with patch.dict(critic_module.os.environ, {"CRITIC_TEST_KEY": "test-token"}), \
             patch.object(critic_module, "urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as call:
            result = endpoint(self.episode())
        request = call.call_args.args[0]
        return result, request, json.loads(request.data), call.call_args.kwargs

    def test_local_chat_request_has_indexed_text_trace_only(self):
        diagnosis = {"frontier": 0, "hint": "Check the requested color."}
        result, request, payload, kwargs = self.call_with_reply(self.reply(json.dumps(diagnosis)))
        self.assertEqual(result, diagnosis)
        self.assertEqual(request.full_url, "http://critic-host:8001/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        self.assertEqual(payload["model"], "separate-large-model")
        self.assertEqual(payload["temperature"], 0)
        trace = json.loads(payload["messages"][1]["content"])["steps"]
        self.assertEqual([row["step"] for row in trace], [0, 1])
        self.assertEqual(set(trace[0]), {"step", "before", "action", "after"})
        self.assertEqual(kwargs["timeout"], 120)

    def test_vllm_uses_generate_and_tokenized_chat_template(self):
        import sys
        from unittest.mock import MagicMock
        config = self.config("vllm")
        settings = config.algorithm.frontier
        settings.critic_enable_thinking = False
        settings.critic_tensor_parallel_size = 2
        settings.critic_gpu_memory_utilization = 0.85
        settings.critic_max_model_len = 16384
        settings.critic_trust_remote_code = False
        vllm = MagicMock()
        vllm.SamplingParams.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
        llm = vllm.LLM.return_value
        llm.llm_engine.model_config.max_model_len = 16384
        llm.get_tokenizer.return_value.apply_chat_template.return_value = [1, 2, 3]
        diagnosis = {"frontier": 0, "hint": "Check color."}
        completion = SimpleNamespace(text=json.dumps(diagnosis), finish_reason="stop")
        llm.generate.return_value = [SimpleNamespace(outputs=[completion])]
        with patch.dict(sys.modules, {"vllm": vllm}):
            critic = critic_module.VLLMFrontierCritic(config)
        self.assertEqual(critic(self.episode()), diagnosis)
        vllm.LLM.assert_called_once_with(model="separate-large-model", tensor_parallel_size=2,
            distributed_executor_backend="mp", gpu_memory_utilization=0.85,
            max_model_len=16384, trust_remote_code=False)
        llm.generate.assert_called_once_with([{"prompt_token_ids": [1, 2, 3]}], critic.sampling, use_tqdm=False)
        template = llm.get_tokenizer.return_value.apply_chat_template.call_args
        self.assertFalse(template.kwargs["enable_thinking"])
        self.assertTrue(template.kwargs["tokenize"])
        trace = json.loads(template.args[0][1]["content"])["steps"]
        self.assertEqual([row["step"] for row in trace], [0, 1])
        self.assertNotIn("input_ids", trace[0])
        critic.enable_thinking = True
        completion.text = 'reasoning</think>\n' + json.dumps(diagnosis)
        self.assertEqual(critic(self.episode()), diagnosis)
        completion.text = '<think>unfinished'
        self.assertIsNone(critic(self.episode()))
        completion.finish_reason = "length"
        self.assertIsNone(critic(self.episode()))
        llm.generate.reset_mock()
        self.assertIsNone(critic({"success": True}))
        llm.generate.assert_not_called()
        llm.llm_engine.model_config.max_model_len = 4
        with self.assertRaisesRegex(ValueError, "context limit"):
            critic(self.episode())

    def test_vllm_worker_reserves_additional_gpus_and_sends_only_text(self):
        import sys
        from unittest.mock import MagicMock
        ray = MagicMock()
        ray.cluster_resources.return_value = {"GPU": 6}
        config = self.config("vllm")
        config.algorithm.frontier.critic_tensor_parallel_size = 2
        config.trainer = SimpleNamespace(n_gpus_per_node=4, nnodes=1)
        with patch.dict(sys.modules, {"ray": ray}):
            critic = critic_module.RayVLLMFrontierCritic(config)
        ray.remote.assert_called_once_with(num_cpus=1, num_gpus=2)
        critic(self.episode())
        trace = critic.actor.diagnose.remote.call_args.args[0]
        self.assertNotIn("input_ids", trace["steps"][0])
        ray.cluster_resources.return_value = {"GPU": 4}
        with patch.dict(sys.modules, {"ray": ray}), self.assertRaisesRegex(ValueError, "additional GPUs"):
            critic_module.RayVLLMFrontierCritic(config)

    def test_openai_responses_request_and_reasoning_output(self):
        diagnosis = {"frontier": 0, "hint": "Check the requested color."}
        result, request, payload, _ = self.call_with_reply(self.reply(json.dumps(diagnosis), "responses"), "responses")
        self.assertEqual(result, diagnosis)
        self.assertTrue(request.full_url.endswith("/responses"))
        self.assertFalse(payload["store"])
        self.assertEqual(payload["max_output_tokens"], 512)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("messages", payload)

    def test_invalid_uncertain_or_truncated_diagnoses_are_rejected(self):
        for api in ("chat_completions", "responses"):
            for text in ('null', 'not JSON', '{"frontier": 9, "hint": "bad index"}',
                         '{"frontier": true, "hint": "boolean index"}',
                         '{"frontier": 0, "hint": " "}',
                         '{"frontier": 0, "hint": "hint", "action": "leaked solution"}'):
                with self.subTest(api=api, text=text):
                    self.assertIsNone(self.call_with_reply(self.reply(text, api), api)[0])
            self.assertIsNone(self.call_with_reply(self.reply('{"frontier": 0, "hint": "hint"}', api, False), api)[0])

    def test_success_is_not_sent_and_local_key_is_optional(self):
        import io
        endpoint = critic_module.ExternalFrontierCritic(self.config())
        with patch.dict(critic_module.os.environ, {}, clear=True), patch.object(critic_module, "urlopen") as call:
            self.assertIsNone(endpoint({"success": True}))
            call.assert_not_called()
            call.return_value = io.BytesIO(json.dumps(self.reply('null')).encode())
            self.assertIsNone(endpoint(self.episode()))
            self.assertIsNone(call.call_args.args[0].get_header("Authorization"))

    def test_endpoint_errors_are_visible_and_hosted_key_required(self):
        endpoint = critic_module.ExternalFrontierCritic(self.config())
        with patch.object(critic_module, "urlopen", side_effect=HTTPError("url", 401, "secret-body", {}, None)):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401") as error:
                endpoint(self.episode())
            self.assertNotIn("secret-body", str(error.exception))
        with patch.dict(critic_module.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "CRITIC_TEST_KEY"):
                critic_module.ExternalFrontierCritic(self.config("responses"))(self.episode())

    def test_ray_runner_receives_named_credential_without_config_value(self):
        import os
        tree = ast.parse((ROOT / "verl/trainer/main_frontier_opsd.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_frontier")
        from unittest.mock import MagicMock
        runner = MagicMock()
        ray = MagicMock()
        ray.is_initialized.return_value = True
        namespace = {"os": os, "ray": ray, "FrontierTaskRunner": runner}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "runner", "exec"), namespace)
        cfg = self.config("responses")
        with patch.dict(os.environ, {"CRITIC_TEST_KEY": "test-token"}):
            namespace["run_frontier"](cfg)
        runner.options.assert_called_once_with(runtime_env={"env_vars": {"CRITIC_TEST_KEY": "test-token"}})
        self.assertNotIn("test-token", repr(cfg))

    def test_trainer_default_uses_external_model_without_actor(self):
        tree = ast.parse((ROOT / "verl/trainer/frontier/trainer.py").read_text())
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_build_critic")
        namespace = {"ExternalFrontierCritic": critic_module.ExternalFrontierCritic}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "critic_builder", "exec"), namespace)
        config = self.config()
        obj = SimpleNamespace(frontier_cfg={"critic_factory": None}, config=config)
        self.assertIsInstance(namespace["_build_critic"](obj), critic_module.ExternalFrontierCritic)


if __name__ == "__main__":
    unittest.main()
