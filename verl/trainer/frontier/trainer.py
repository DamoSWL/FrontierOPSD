"""Progressive frontier self-distillation using SDAR data, workers and validation."""

import importlib
from pathlib import Path
from frontier_opsd.records import RolloutRecords
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.frontier_runtime import FrontierRuntime
from verl.trainer.frontier.critic import ExternalFrontierCritic, RayVLLMFrontierCritic, validate_diagnosis
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.metric import reduce_metrics


class FrontierOPSDTrainer(FrontierRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frontier_cfg = self.config.algorithm.frontier
        self.policy_version = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.task_schedule = {}
        self.records = RolloutRecords(Path(self.config.trainer.default_local_dir) / "rollouts.jsonl")

    def _build_critic(self):
        factory_name = self.frontier_cfg.get("critic_factory")
        if factory_name:
            module, name = factory_name.split(":")
            return getattr(importlib.import_module(module), name)(
                self.config, self.tokenizer, self.actor_rollout_wg)
        if self.frontier_cfg.get("critic_api") == "vllm":
            return RayVLLMFrontierCritic(self.config)
        return ExternalFrontierCritic(self.config)

    def _diagnose(self, episodes, critic):
        return [None if episode["success"] else validate_diagnosis(critic(episode), episode)
                for episode in episodes]

    def _training_batch(self, newest):
        if not newest:
            raise ValueError("A frontier KL update requires diagnosed current-round targets")
        if any(row["policy_version"] != self.policy_version for row in newest):
            raise ValueError("Frontier KL targets must come from the current policy round")
        rows = newest
        fields = ("input_ids", "attention_mask", "position_ids", "responses", "response_mask", "frontier_teacher_log_probs", "old_log_probs")
        batch = DataProto.from_single_dict(collate_fn(
            [{key: row[key] for key in fields} for row in rows]))
        weights = [1 / len(rows)] * len(rows)
        batch.batch["frontier_weights"] = torch.tensor(weights, dtype=torch.float32)
        # Padding carries zero weight, so repeated examples cannot alter
        # the current-round mean KL, even with only one frontier example.
        padded, padding = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
        if padding:
            padded.batch["frontier_weights"][-padding:] = 0
        padded.meta_info.update(temperature=1.0, frontier_dp_size=self.actor_rollout_wg.world_size,
            global_token_num=padded.batch["attention_mask"].sum(-1).tolist())
        return padded

    def _cache_targets(self, newest):
        # Both scoring passes finish before any update. The same current actor
        # is the frozen self-teacher; only its context differs.
        for row in newest:
            for privileged, field in ((False, "old_log_probs"), (True, "frontier_teacher_log_probs")):
                prefix = "teacher_" if privileged else ""
                batch = DataProto.from_single_dict(collate_fn([{
                    key: row[prefix + key] for key in ("input_ids", "attention_mask", "position_ids")
                } | {"responses": row["responses"]}]))
                padded, _ = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
                output = self.actor_rollout_wg.compute_frontier_teacher(padded)
                row[field] = output.batch["frontier_teacher_log_probs"][0].detach().cpu().clone()
            for key in ("teacher_input_ids", "teacher_attention_mask", "teacher_position_ids"):
                row.pop(key, None)

    def _save_checkpoint(self):
        folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        folder.mkdir(parents=True, exist_ok=True)
        remote_root = self.config.trainer.get("default_hdfs_dir")
        if remote_root:
            raise NotImplementedError("Frontier checkpoints currently use local storage")
        self.actor_rollout_wg.save_checkpoint(str(folder / "actor"), None, self.global_steps,
                                           self.config.trainer.get("max_actor_ckpt_to_keep"))
        torch.save(self.train_dataloader.state_dict(), folder / "data.pt")
        epoch_complete = self.batch_in_epoch >= len(self.train_dataloader)
        torch.save({"policy_version": self.policy_version,
                    "epoch": self.epoch + int(epoch_complete),
                    "batch_in_epoch": 0 if epoch_complete else self.batch_in_epoch, "task_schedule": self.task_schedule,
                    "latest_records": self.records.latest,
                    "record_offset": self.records.path.stat().st_size if self.records.path.exists() else 0}, folder / "frontier.pt")
        tracker = Path(self.config.trainer.default_local_dir) / "latest_checkpointed_iteration.txt"
        temporary = tracker.with_suffix(".tmp")
        temporary.write_text(str(self.global_steps))
        temporary.replace(tracker)

    def _load_checkpoint(self):
        mode = self.config.trainer.resume_mode
        if mode == "disable":
            return
        if mode == "resume_path":
            folder = Path(self.config.trainer.resume_from_path)
        elif mode == "auto":
            tracker = Path(self.config.trainer.default_local_dir) / "latest_checkpointed_iteration.txt"
            if not tracker.exists():
                return
            folder = tracker.parent / f"global_step_{int(tracker.read_text().strip())}"
        else:
            raise ValueError(f"Unsupported frontier resume mode: {mode}")
        if not (folder / "frontier.pt").exists() or not (folder / "data.pt").exists():
            raise RuntimeError(f"Frontier checkpoint requires policy-round and dataloader state: {folder}")
        self.actor_rollout_wg.load_checkpoint(str(folder / "actor"),
            del_local_after_load=self.config.trainer.get("del_local_ckpt_after_load", False))
        self.train_dataloader.load_state_dict(torch.load(folder / "data.pt", weights_only=False))
        state = torch.load(folder / "frontier.pt", map_location="cpu", weights_only=True)
        self.policy_version = state["policy_version"]
        self.epoch = state.get("epoch", 0)
        self.batch_in_epoch = state.get("batch_in_epoch", 0)
        self.task_schedule = state.get("task_schedule", {})
        self.records.latest = state.get("latest_records", {})
        if "record_offset" in state:
            self.records.rewind(state["record_offset"])
        self.global_steps = int(folder.name.split("global_step_")[-1])

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True))
        self.global_steps = 0
        self._load_checkpoint()
        if self.config.trainer.get("val_before_train", True):
            logger.log(self._validate(), step=self.global_steps)
        if self.config.trainer.get("val_only", False):
            return
        critic = self._build_critic()
        while self.epoch < self.config.trainer.total_epochs:
            for batch_dict in self.train_dataloader:
                if self.global_steps >= self.total_training_steps:
                    return
                original = DataProto.from_single_dict(batch_dict)
                # One unhinted rollout per task; no same-batch repeat rollout.
                real_count = len(original)
                gen_batch, _ = pad_dataproto_to_divisor(original,
                    self.config.data.train_batch_size)
                slot = self.batch_in_epoch
                episodes = self.traj_collector.collect(gen_batch, self.actor_rollout_wg, self.envs,
                    task_specs=self.task_schedule.get(slot), task_count=real_count,
                    excluded_task_ids=[spec["identity"] for specs in self.task_schedule.values()
                                       for spec in specs])
                if slot not in self.task_schedule:
                    self.task_schedule[slot] = [episode["task_spec"] for episode in episodes]
                episodes = episodes[:real_count]
                diagnoses = self._diagnose(episodes, critic)
                newest = []
                comparisons = {key: 0 for key in ("improved", "regressed", "unchanged", "uncertain")}
                for episode, diagnosis in zip(episodes, diagnoses):
                    record = {"task_id": episode["task_id"], "epoch": self.epoch + 1,
                        "policy_version": self.policy_version, "success": episode["success"],
                        "task": episode["task_spec"],
                        "steps": [{"step_index": index, "observation_before": row["frontier_before"],
                            "response": row["frontier_action"], "action": row["frontier_action"],
                            "observation_after": row["frontier_after"],
                            "response_token_ids": row["responses"].tolist()}
                            for index, row in enumerate(episode["steps"])],
                        "diagnosis": None if diagnosis is None else {
                            "mistaken_step_index": diagnosis["frontier"], "hint": diagnosis["hint"]},
                        "progress": {}, "training_status": "success_skipped" if episode["success"]
                            else "undiagnosed" if diagnosis is None else "distillation"}
                    # History is analysis-only; targets always come from episode.
                    self.records.add(record)
                    comparisons[record["comparison"]["result"]] += 1
                    if diagnosis is None:
                        continue
                    row = self.traj_collector.scoring_target(
                        episode["steps"][diagnosis["frontier"]], diagnosis["hint"])
                    if row["response_mask"].sum() == 0:
                        continue
                    row["policy_version"] = self.policy_version
                    newest.append(row)
                metrics = {"training/epoch": self.epoch + 1,
                    "frontier/success_rate": sum(ep["success"] for ep in episodes) / len(episodes),
                    "frontier/training_examples": len(newest)}
                metrics.update({"frontier/" + key: value for key, value in comparisons.items()})
                if newest:
                    self._cache_targets(newest)
                    metrics.update(reduce_metrics(self.actor_rollout_wg.update_frontier_actor(
                        self._training_batch(newest)).meta_info["metrics"]))
                    self.policy_version += 1
                    self.global_steps += 1
                self.batch_in_epoch += 1
                metrics.update({"training/global_step": self.global_steps,
                                "frontier/policy_version": self.policy_version})
                last = self.global_steps >= self.total_training_steps
                if newest and self.config.trainer.test_freq > 0 and (last or self.global_steps % self.config.trainer.test_freq == 0):
                    metrics.update(self._validate())
                if newest and self.config.trainer.save_freq > 0 and (last or self.global_steps % self.config.trainer.save_freq == 0):
                    self._save_checkpoint()
                logger.log(metrics, step=self.global_steps)
                del newest
                if last:
                    return
            self.epoch += 1
            self.batch_in_epoch = 0
