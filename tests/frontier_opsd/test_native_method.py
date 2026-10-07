"""Exercise actual SDAR method code without importing Ray/CUDA dependencies.

AST loading isolates methods from dependency-heavy modules. Tensor operations
and optimizer updates use real PyTorch; only orchestration objects are fakes.
"""

import ast
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def load_class(path, name, methods, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    node.bases = []
    node.decorator_list = []
    node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in methods]
    for item in node.body:
        item.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[name]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class NativeMethodTests(unittest.TestCase):
    def test_training_batch_averages_only_current_policy_targets(self):
        class Proto:
            def __init__(self, batch):
                self.batch, self.meta_info = batch, {}
            @classmethod
            def from_single_dict(cls, batch):
                return cls(batch)

        def collate(rows):
            return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}

        def pad(proto, divisor):
            count = len(proto.batch["responses"])
            padding = (-count) % divisor
            result = Proto({key: torch.cat([value, value[:1].repeat(padding, *([1] * (value.ndim - 1)))])
                            for key, value in proto.batch.items()})
            return result, padding

        Trainer = load_class("verl/trainer/frontier/trainer.py", "FrontierOPSDTrainer", {"_training_batch"},
            {"torch": torch, "DataProto": Proto, "collate_fn": collate, "pad_dataproto_to_divisor": pad})
        trainer = Trainer()
        trainer.policy_version = 3
        trainer.actor_rollout_wg = SimpleNamespace(world_size=4)
        row = {"policy_version": 3, "input_ids": torch.tensor([0, 1]),
               "attention_mask": torch.ones(2), "position_ids": torch.arange(2),
               "responses": torch.tensor([1]), "response_mask": torch.ones(1),
               "frontier_teacher_log_probs": torch.tensor([.7]).log(),
               "old_log_probs": torch.tensor([.3]).log()}
        batch = trainer._training_batch([deepcopy(row), deepcopy(row)])
        torch.testing.assert_close(batch.batch["frontier_weights"], torch.tensor([.5, .5, 0., 0.]))
        self.assertEqual(batch.meta_info["frontier_dp_size"], 4)
        stale = {**row, "policy_version": 2}
        with self.assertRaisesRegex(ValueError, "current policy round"):
            trainer._training_batch([row, stale])
        with self.assertRaisesRegex(ValueError, "current-round"):
            trainer._training_batch([])

    def test_cache_scores_same_tokens_in_both_contexts_before_update(self):
        class Proto:
            def __init__(self, batch):
                self.batch = batch
            @classmethod
            def from_single_dict(cls, batch):
                return cls(batch)
        def collate(rows):
            return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
        calls = []
        class Worker:
            world_size = 1
            def compute_frontier_teacher(self, batch):
                calls.append(deepcopy(batch.batch))
                return Proto({"frontier_teacher_log_probs": batch.batch["input_ids"][:, :1].float()})
        Trainer = load_class("verl/trainer/frontier/trainer.py", "FrontierOPSDTrainer", {"_cache_targets"},
            {"DataProto": Proto, "collate_fn": collate, "pad_dataproto_to_divisor": lambda b, _: (b, 0)})
        trainer = Trainer()
        trainer.actor_rollout_wg = Worker()
        row = {"responses": torch.tensor([2]), "input_ids": torch.tensor([0, 2]),
            "attention_mask": torch.ones(2), "position_ids": torch.arange(2),
            "teacher_input_ids": torch.tensor([9, 0, 2]), "teacher_attention_mask": torch.ones(3),
            "teacher_position_ids": torch.arange(3)}
        trainer._cache_targets([row])
        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(calls[0]["responses"], calls[1]["responses"])
        torch.testing.assert_close(row["old_log_probs"], torch.tensor([0.]))
        torch.testing.assert_close(row["frontier_teacher_log_probs"], torch.tensor([9.]))
        self.assertNotIn("teacher_input_ids", row)
        self.assertFalse(row["old_log_probs"].requires_grad)
        self.assertFalse(row["frontier_teacher_log_probs"].requires_grad)

    def test_actual_fit_one_rollout_per_batch_per_epoch_uses_fresh_tokens(self):
        from types import ModuleType
        from frontier_opsd.records import RolloutRecords
        class Proto:
            @classmethod
            def from_single_dict(cls, batch):
                proto = cls()
                proto.identity = batch["identity"]
                return proto
            def __len__(self):
                return 1
        trace = []
        class Collector:
            visits = 0
            targets = []
            def collect(self, gen, actor, envs, task_specs=None, **kwargs):
                self.visits += 1
                trace.append(("rollout", gen.identity))
                spec = {"identity": gen.identity}
                if task_specs:
                    self_test.assertEqual(task_specs, [spec])
                token = 7 if self.visits < 3 else 8
                return [{"success": gen.identity == "B", "task_id": gen.identity, "task_spec": spec,
                    "steps": [{"responses": torch.tensor([token]), "frontier_before": "before",
                               "frontier_action": str(token), "frontier_after": "after"}]}]
            def scoring_target(self, original, hint):
                self.targets.append(original["responses"].item())
                return {**original, "response_mask": torch.ones(1)}
        class Actor:
            def update_frontier_actor(self, batch):
                trace.append(("update", batch[0]["responses"].item()))
                return SimpleNamespace(meta_info={"metrics": {}})
        omega = ModuleType("omegaconf")
        omega.OmegaConf = SimpleNamespace(to_container=lambda cfg, **kwargs: {})
        tracking = ModuleType("verl.utils.tracking")
        tracking.Tracking = lambda **kwargs: SimpleNamespace(log=lambda *args, **kwargs: None)
        Trainer = load_class("verl/trainer/frontier/trainer.py", "FrontierOPSDTrainer", {"fit"},
            {"DataProto": Proto, "reduce_metrics": lambda metrics: metrics,
             "pad_dataproto_to_divisor": lambda batch, _: (batch, 0)})
        self_test = self
        trainer = Trainer()
        trainer.config = Config(trainer=Config(project_name="test", experiment_name="test", logger=[],
            val_before_train=False, val_only=False, total_epochs=2, test_freq=0, save_freq=0),
            data=Config(train_batch_size=1), env=Config(rollout=Config(n=1)))
        trainer.epoch, trainer.batch_in_epoch, trainer.policy_version = 0, 0, 0
        trainer.task_schedule = {}
        trainer.records = RolloutRecords()
        trainer.total_training_steps = 4
        trainer.train_dataloader = [{"identity": "A"}, {"identity": "B"}]
        trainer.envs = object()
        trainer.actor_rollout_wg = Actor()
        trainer.traj_collector = Collector()
        trainer._load_checkpoint = lambda: None
        trainer._build_critic = lambda: object()
        trainer._diagnose = lambda episodes, critic: [None if ep["success"] else {"frontier": 0, "hint": "hint"} for ep in episodes]
        trainer._cache_targets = lambda rows: None
        trainer._training_batch = lambda rows: rows
        with patch.dict("sys.modules", {"omegaconf": omega, "verl.utils.tracking": tracking}):
            trainer.fit()
        self.assertEqual(trace, [("rollout", "A"), ("update", 7), ("rollout", "B"),
                                ("rollout", "A"), ("update", 8), ("rollout", "B")])
        self.assertEqual(trainer.traj_collector.targets, [7, 8])
        self.assertEqual(trainer.records.latest["A"]["epoch"], 2)
        self.assertEqual(trainer.records.latest["A"]["comparison"]["previous_epoch"], 1)
        self.assertEqual(trainer.records.latest["B"]["training_status"], "success_skipped")
        self.assertEqual(trainer.policy_version, 2)

    def test_runtime_initializes_without_rl_configuration(self):
        class Loader:
            def __init__(self, dataset, **kwargs):
                self.dataset, self.kwargs = dataset, kwargs
            def __len__(self):
                return 2
        Runtime = load_class("verl/trainer/frontier_runtime.py", "FrontierRuntime", {"__init__"},
            {"StatefulDataLoader": Loader, "open_dict": lambda _: nullcontext(),
             "ValidationGenerationsLogger": lambda: object()})
        config = Config(data=Config(train_batch_size=2, val_batch_size=2, dataloader_num_workers=0),
            trainer=Config(total_epochs=3), actor_rollout_ref=Config(actor=Config(optim=Config())))
        runtime = Runtime(config, None, None, [1, 2], [3, 4], None, None, None, None, None, None)
        self.assertEqual(runtime.total_training_steps, 6)
        self.assertIs(runtime.train_dataloader.dataset[0], 1)
        self.assertEqual(config.actor_rollout_ref.actor.optim.total_training_steps, 6)
        self.assertNotIn("algorithm", config)
        self.assertNotIn("critic", config)
        self.assertNotIn("reward_model", config)

    def test_frontier_path_has_no_rl_trainer_or_update_dispatch(self):
        path = ROOT / "verl/trainer/frontier/trainer.py"
        tree = ast.parse(path.read_text())
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FrontierOPSDTrainer")
        self.assertEqual([base.id for base in trainer.bases], ["FrontierRuntime"])
        calls = {node.func.attr for node in ast.walk(trainer) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertIn("update_frontier_actor", calls)
        self.assertNotIn("update_actor", calls)
        self.assertNotIn("update_opd_actor", calls)
        for relative in ("verl/trainer/main_frontier_opsd.py", "verl/trainer/frontier_runtime.py",
                         "verl/trainer/frontier_data.py", "verl/trainer/frontier/trainer.py"):
            source = (ROOT / relative).read_text()
            self.assertNotIn("RayPPOTrainer", source)
            self.assertNotIn("run_sdar", source)
            self.assertNotIn("compute_advantage", source)
            self.assertNotIn("compute_policy_loss", source)

    def test_actual_collector_preserves_original_tokens_and_hint_isolation(self):
        import numpy as np
        from types import ModuleType

        class Proto:
            def __init__(self, batch, non=None):
                self.batch, self.non_tensor_batch, self.meta_info = batch, non or {}, {}
            def __len__(self):
                return len(self.batch["input_ids"])
            def select(self, keys, non_keys):
                return Proto({key: self.batch[key].clone() for key in keys},
                             {key: deepcopy(self.non_tensor_batch[key]) for key in non_keys})
            def pop(self, batch_keys):
                for key in batch_keys:
                    self.batch.pop(key)
            def union(self, other):
                for key, value in other.non_tensor_batch.items():
                    if key in self.non_tensor_batch and not np.array_equal(self.non_tensor_batch[key], value):
                        raise ValueError("Metadata conflict")
                self.batch.update(other.batch)
                self.non_tensor_batch.update(other.non_tensor_batch)
                return self

        def preprocess(gen, obs):
            text = obs["text"][0]
            token = 9 if "Privileged" in text else int(text.split(":")[-1])
            return Proto({"input_ids": torch.tensor([[token]]), "attention_mask": torch.ones(1, 1, dtype=torch.long),
                          "position_ids": torch.zeros(1, 1, dtype=torch.long)},
                         {"raw_prompt_ids": np.array([[token]], dtype=object),
                          "raw_prompt": np.array([text], dtype=object)})

        def generate(actor, prompts):
            ids = torch.cat((prompts.batch["input_ids"], torch.tensor([[2]])), -1)
            return Proto({"input_ids": ids, "responses": torch.tensor([[2]]),
                          "prompts": prompts.batch["input_ids"].clone(),
                          "attention_mask": torch.ones(1, 2, dtype=torch.long),
                          "position_ids": torch.tensor([[0, 1]])}, deepcopy(prompts.non_tensor_batch))

        def rows(proto):
            return [{**{key: value[0] for key, value in proto.batch.items()},
                     **{key: value[0] for key, value in proto.non_tensor_batch.items()}}]

        class Environment:
            def reset(self, kwargs):
                self.position = 0
                return {"text": ["state:0"], "image": None}, [{}]
            def step(self, actions):
                self.position += 1
                return {"text": [f"state:{self.position}"], "image": None}, np.array([0]), np.array([self.position == 2]), [{}]
            def success_evaluator(self, **kwargs):
                return {"success_rate": np.array([True])}

        namespace = {"deepcopy": deepcopy, "np": np, "torch": torch, "generate": generate,
                     "_gamefiles_from_infos": lambda infos: np.array([None], dtype=object),
                     "to_list_of_dict": rows, "json": __import__("json"),
                     "task_id": __import__("frontier_opsd.records", fromlist=["task_id"]).task_id}
        Collector = load_class("verl/trainer/frontier/rollout.py", "FrontierTrajectoryCollector", {"collect", "scoring_target"}, namespace)
        collector = Collector()
        collector.config = Config(env=Config(env_name="search", max_steps=2),
                                  data=Config(max_prompt_length=8))
        collector.preprocess_batch = preprocess
        collector.tokenizer = SimpleNamespace(batch_decode=lambda *args, **kwargs: ["action"],
                                              encode=lambda *args, **kwargs: [9])
        model_module = ModuleType("verl.utils.model")
        model_module.compute_position_id_with_mask = lambda mask: mask.cumsum(-1) - 1
        plan = {"frontier": 0, "hint": "reconsider", "episode": {"steps": [{"frontier_before": "state:0"}]}}
        with patch.dict("sys.modules", {"verl.utils.model": model_module}):
            episode = collector.collect(Proto({"input_ids": torch.tensor([[0]])}), None,
                                        Environment(), repeat_tasks=True)[0]
            target = collector.scoring_target(episode["steps"][0], "reconsider")
        self.assertEqual(target["teacher_input_ids"][0].item(), 9)
        self.assertEqual(target["input_ids"][0].item(), 0)
        self.assertEqual(target["raw_prompt"], "state:0")
        self.assertEqual(target["raw_prompt_ids"].tolist(), [0])
        self.assertFalse(target["frontier_target"])
        torch.testing.assert_close(target["responses"], episode["steps"][0]["responses"])
        torch.testing.assert_close(target["teacher_input_ids"], torch.tensor([9, 0, 2]))
        collector.config.data.max_prompt_length = 1
        with self.assertRaisesRegex(ValueError, "exceeds"):
            collector.scoring_target(episode["steps"][0], "reconsider")
        self.assertFalse(episode["steps"][1]["frontier_target"])

    def test_actual_actor_opsd_gradients_with_padding(self):
        class TensorBatch(dict):
            def split(self, count):
                size = len(self["responses"])
                return [TensorBatch({key: value[start:start + count] for key, value in self.items()})
                        for start in range(0, size, count)]
            def to(self, device):
                return TensorBatch({key: value.to(device) for key, value in self.items()})

        class Proto:
            def __init__(self, values):
                self.batch = TensorBatch(values)
                self.meta_info = {"frontier_dp_size": 2}
            def select(self, batch_keys):
                return Proto({key: self.batch[key] for key in batch_keys})

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.table = torch.nn.Parameter(torch.tensor([[1., 2., 0.], [3., 0., 1.], [0., 0., 0.]]))
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.table[input_ids])

        def append(metrics, values):
            for key, value in values.items():
                metrics.setdefault(key, []).append(value)

        namespace = {"torch": torch, "DataProto": Proto, "append_to_dict": append,
                     "get_torch_device": lambda: SimpleNamespace(current_device=lambda: "cpu")}
        Actor = load_class("verl/workers/actor/dp_actor.py", "DataParallelPPOActor",
                           {"update_frontier_policy"}, namespace)
        for objective in ("sampled_opsd",):
            with self.subTest(objective=objective):
                actor = Actor()
                actor.actor_module = Model()
                actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=.1)
                actor.config = Config(frontier_micro_batch_size_per_gpu=1, frontier_objective=objective)
                actor.device_name, actor.use_ulysses_sp = "cpu", False
                def optimizer_step():
                    norm = torch.nn.utils.clip_grad_norm_(actor.actor_module.parameters(), 100)
                    actor.actor_optimizer.step()
                    return norm
                actor._optimizer_step = optimizer_step
                actor._forward_micro_batch = lambda micro, **kwargs: (None,
                    actor.actor_module(input_ids=micro["input_ids"]).logits[:, :-1].log_softmax(-1).gather(-1, micro["responses"].unsqueeze(-1)).squeeze(-1))
                q = torch.tensor([[-.3, -1.], [-.4, -.9]])
                q.requires_grad_(True)
                values = {"input_ids": torch.tensor([[0, 2, 2], [1, 2, 2]]),
                    "attention_mask": torch.ones(2, 3), "position_ids": torch.arange(3).repeat(2, 1),
                    "responses": torch.tensor([[2, 2], [2, 2]]), "response_mask": torch.ones(2, 2),
                    "frontier_weights": torch.tensor([1., 0.]), "frontier_teacher_log_probs": q,
                    "old_log_probs": torch.tensor([[-1.2, -1.1], [-1.3, -1.1]])}
                before = actor.actor_module.table.detach().clone()
                expected_table = before.clone().requires_grad_(True)
                log_p = expected_table[values["input_ids"]][0, :-1].log_softmax(-1)
                from verl.trainer.ppo.skillsd_utils import compute_sdl_loss
                sampled = log_p.gather(-1, values["responses"][0].unsqueeze(-1)).squeeze(-1)
                expected_loss = compute_sdl_loss(sampled.unsqueeze(0), q[:1],
                    values["old_log_probs"][:1], torch.ones(1, 2)) * 2
                expected_loss.backward()
                metrics = actor.update_frontier_policy(Proto(values))
                torch.testing.assert_close(actor.actor_module.table, before - .1 * expected_table.grad)
                self.assertAlmostEqual(metrics["frontier/loss"][0], expected_loss.item(), places=5)
                self.assertIsNone(q.grad)

    def test_same_task_alfworld_reset_preserves_sampler(self):
        class Environment:
            def __init__(self, games):
                self.games = games
                self.next = 0
                self.closed = False
            def seed(self, seed):
                pass
            def reset(self):
                game = self.games[self.next % len(self.games)]
                self.next += 1
                return [game], {"extra.gamefile": [game]}
            def close(self):
                self.closed = True

        class Base:
            game_files = ["train/gameA", "train/gameB"]
            def init_env(self, batch_size):
                return Environment(list(self.game_files))

        Worker = load_class("agent_system/environments/env_package/alfworld/envs.py", "AlfworldWorker",
                            {"__init__", "reset"}, {"deepcopy": deepcopy})
        worker = Worker({}, 0, Base())
        self.assertEqual(worker.reset()[0], ["train/gameA"])
        self.assertEqual(worker.reset(repeat_task=True)[0], ["train/gameA"])
        self.assertEqual(worker.reset(repeat_task=True)[0], ["train/gameA"])
        pinned = worker.repair_env
        self.assertEqual(worker.reset()[0], ["train/gameB"])
        self.assertTrue(pinned.closed)
        self.assertEqual(worker.reset(repeat_task=True)[0], ["train/gameB"])
        self.assertEqual(worker.reset(gamefile="train/gameA")[0], ["train/gameA"])
        self.assertEqual(worker.reset(gamefile="train/gameB")[0], ["train/gameB"])

    def test_webshop_epoch_task_selection_without_replacement_and_explicit_reset(self):
        import numpy as np
        class Worker:
            def __init__(self):
                self.reset = SimpleNamespace(remote=lambda idx: (str(idx), {}))
        Envs = load_class("agent_system/environments/env_package/webshop/envs.py", "WebshopMultiProcessEnv",
                          {"reset"}, {"np": np, "ray": SimpleNamespace(get=lambda futures: futures)})
        env = Envs()
        env._workers = [Worker(), Worker()]
        env.num_processes = env.env_num = 2
        env.group_n = 1
        env._frontier_pool = [500, 501, 502]
        env._frontier_cursor = 0
        env._rng = np.random.RandomState(0)
        first, infos = env.reset(task_ids="next", task_count=2)
        self.assertEqual(first, ["500", "501"])
        self.assertEqual([info["frontier_session_id"] for info in infos], [500, 501])
        second, _ = env.reset(task_ids="next", task_count=1, excluded_task_ids=[500, 501])
        self.assertEqual(second, ["502", "502"])
        again, _ = env.reset(task_ids=[500, 501])
        self.assertEqual(again, first)
        with self.assertRaisesRegex(ValueError, "task count"):
            env.reset(task_ids="next", excluded_task_ids=[500, 501, 502])

    def test_frontier_configuration_preserves_data_and_disables_rl(self):
        tree = ast.parse((ROOT / "verl/trainer/main_frontier_opsd.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "configure_frontier")
        namespace = {"math": __import__("math"), "open_dict": lambda _: nullcontext()}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "config", "exec"), namespace)
        data = Config(train_files="same/train.parquet", val_files="same/test.parquet", train_batch_size=16)
        cfg = Config(data=data, algorithm=Config(filter_groups=Config(enable=True), adv_estimator="grpo", frontier=Config(critic_model="separate-test-critic")), reward_model=Config(enable=True),
            env=Config(env_name="alfworld/AlfredTWEnv", seed=0, rollout=Config(n=1)),
            actor_rollout_ref=Config(actor=Config(strategy="fsdp", use_kl_loss=True, ppo_micro_batch_size_per_gpu=1), model=Config(),
                                    rollout=Config(mode="sync", name="vllm", n=1)))
        namespace["configure_frontier"](cfg)
        self.assertIs(cfg.data, data)
        self.assertEqual(dict(cfg.data), {"train_files": "same/train.parquet", "val_files": "same/test.parquet", "train_batch_size": 16})
        self.assertEqual(cfg.env.rollout.n, 1)
        self.assertNotIn("adv_estimator", cfg.algorithm)
        self.assertFalse(cfg.actor_rollout_ref.actor.use_kl_loss)
        self.assertFalse(cfg.actor_rollout_ref.actor.use_sdar_loss)
        self.assertFalse(cfg.reward_model.enable)
        self.assertEqual(cfg.actor_rollout_ref.actor.frontier_objective, "sampled_opsd")
        cfg.algorithm.frontier["critic_model"] = None
        with self.assertRaisesRegex(ValueError, "separate critic model"):
            namespace["configure_frontier"](cfg)
        cfg.algorithm.frontier["critic_model"] = "separate-test-critic"
        cfg.algorithm.frontier["objective"] = "sampled_ce"
        with self.assertRaisesRegex(ValueError, "sole training loss"):
            namespace["configure_frontier"](cfg)
        cfg.algorithm.frontier["objective"] = "sampled_opsd"
        cfg.algorithm.frontier["replay_capacity"] = 0
        with self.assertRaisesRegex(ValueError, "replay has been removed"):
            namespace["configure_frontier"](cfg)


if __name__ == "__main__":
    unittest.main()
