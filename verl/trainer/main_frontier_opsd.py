"""FrontierOPSD entry point; reuse SDAR's datasets and environment setup."""

import os

import hydra
import ray
from omegaconf import OmegaConf, open_dict


def configure_frontier(config):
    defaults = {"objective": "sampled_opsd", "update_steps": 1, "critic_factory": None,
                "critic_model": None, "critic_api": "vllm", "critic_base_url": "http://127.0.0.1:8001/v1",
                "critic_api_key_env": "FRONTIER_CRITIC_API_KEY",
                "critic_timeout": 120, "critic_max_tokens": 512,
                "critic_enable_thinking": False, "critic_tensor_parallel_size": 2,
                "critic_gpu_memory_utilization": 0.85, "critic_max_model_len": 16384,
                "critic_trust_remote_code": False}
    supplied = config.algorithm.get("frontier", {})
    if "max_rounds" in supplied:
        raise ValueError("Per-batch rounds have been removed; use trainer.total_epochs")
    if config.data.get("gen_batch_size", config.data.train_batch_size) > config.data.train_batch_size:
        raise ValueError("gen_batch_size cannot exceed the fixed environment worker batch size")
    if config.env.rollout.n != 1:
        raise ValueError("Epoch-based FrontierOPSD requires env.rollout.n=1")
    if "candidates" in supplied:
        raise ValueError("Repair candidates have been removed; omit algorithm.frontier.candidates")
    if "replay_capacity" in supplied or "replay_weight" in supplied:
        raise ValueError("Historical anchor replay has been removed; omit replay_capacity and replay_weight")
    values = {key: supplied.get(key, value) for key, value in defaults.items()}
    if values["objective"] != "sampled_opsd":
        raise ValueError("FrontierOPSD supports only sampled_opsd; KL is the sole training loss")
    for key in ("update_steps",):
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError(f"algorithm.frontier.{key} must be a positive integer")
    if not values["critic_factory"]:
        if values["critic_api"] not in ("vllm", "chat_completions", "responses"):
            raise ValueError("critic_api must be vllm, chat_completions or responses")
        if type(values["critic_enable_thinking"]) is not bool:
            raise ValueError("critic_enable_thinking must be a boolean")
        if not isinstance(values["critic_model"], str) or not values["critic_model"].strip():
            raise ValueError("Set +algorithm.frontier.critic_model to a separate critic model, or supply critic_factory")
        if values["critic_api"] != "vllm" and (not isinstance(values["critic_base_url"], str) or not values["critic_base_url"].startswith(("http://", "https://"))):
            raise ValueError("critic_base_url must be an HTTP(S) API base URL")
        for key in ("critic_timeout", "critic_max_tokens", "critic_tensor_parallel_size", "critic_max_model_len"):
            if type(values[key]) is not int or values[key] < 1:
                raise ValueError(f"algorithm.frontier.{key} must be a positive integer")
        if type(values["critic_trust_remote_code"]) is not bool:
            raise ValueError("critic_trust_remote_code must be a boolean")
        memory = values["critic_gpu_memory_utilization"]
        if isinstance(memory, bool) or not isinstance(memory, (int, float)) or not 0 < memory <= 1:
            raise ValueError("critic_gpu_memory_utilization must be in (0, 1]")
        if values["critic_max_tokens"] >= values["critic_max_model_len"]:
            raise ValueError("critic_max_model_len must leave room for prompt tokens")
        if not isinstance(values["critic_api_key_env"], str) or not values["critic_api_key_env"]:
            raise ValueError("critic_api_key_env must name an environment variable")
    name = config.env.env_name.lower()
    if not any(env in name for env in ("alfworld", "webshop", "search")):
        raise NotImplementedError("FrontierOPSD uses SDAR's ALFWorld, WebShop and Search environments")
    if config.actor_rollout_ref.actor.strategy not in ("fsdp", "fsdp2"):
        raise NotImplementedError("FrontierOPSD currently requires FSDP/FSDP2")
    if config.actor_rollout_ref.rollout.mode != "sync" or config.actor_rollout_ref.rollout.name != "vllm":
        raise NotImplementedError("FrontierOPSD currently requires synchronous vLLM rollout")
    if config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) != 1:
        raise NotImplementedError("FrontierOPSD currently requires sequence parallel size 1")
    if config.actor_rollout_ref.rollout.n != 1:
        raise ValueError("Keep rollout.n=1; SDAR environment grouping uses env.rollout.n")
    with open_dict(config):
        config.algorithm.frontier = values
        config.algorithm.pop("adv_estimator", None)
        config.algorithm.use_kl_in_reward = False
        config.algorithm.filter_groups.enable = False
        config.reward_model.enable = False
        actor = config.actor_rollout_ref.actor
        actor.use_frontier_loss = True
        actor.frontier_objective = values["objective"]
        actor.frontier_update_steps = values["update_steps"]
        actor.frontier_micro_batch_size_per_gpu = actor.get("frontier_micro_batch_size_per_gpu", actor.get("ppo_micro_batch_size_per_gpu"))
        if type(actor.frontier_micro_batch_size_per_gpu) is not int or actor.frontier_micro_batch_size_per_gpu < 1:
            raise ValueError("Set a positive frontier_micro_batch_size_per_gpu or SDAR microbatch size")
        actor.use_sdar_loss = False
        actor.use_sdl_loss = False
        actor.use_kl_loss = False
        actor.use_invalid_action_penalty = False
        actor.use_dynamic_bsz = False
        # Scoring uses sampled token log-probabilities, not dense vocabulary targets.
        config.actor_rollout_ref.rollout.temperature = 1.0
        config.actor_rollout_ref.rollout.top_p = 1.0
        config.actor_rollout_ref.rollout.top_k = -1
        config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz = False
    return config


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_frontier(configure_frontier(config))


def run_frontier(config):
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
        kwargs = OmegaConf.to_container(OmegaConf.create(config.get("ray_init", {})), resolve=True)
        kwargs["runtime_env"] = OmegaConf.to_container(OmegaConf.merge(
            get_ppo_ray_runtime_env(), kwargs.get("runtime_env", {})), resolve=True)
        ray.init(**kwargs)
    key_name = config.algorithm.frontier.critic_api_key_env
    critic_env = {key_name: os.environ[key_name]} if key_name in os.environ else {}
    runner = FrontierTaskRunner.options(runtime_env={"env_vars": critic_env})
    ray.get(runner.remote().run.remote(config))


@ray.remote(num_cpus=1)
class FrontierTaskRunner:
    def run(self, config):
        from agent_system.environments import make_envs
        from agent_system.reward_manager import EpisodeRewardManager
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.frontier_data import create_rl_dataset, create_rl_sampler
        from verl.trainer.frontier.rollout import FrontierTrajectoryCollector
        from verl.trainer.frontier.trainer import FrontierOPSDTrainer
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        OmegaConf.resolve(config)
        local_path = copy_to_local(config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False))
        trust = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust)
        processor = hf_processor(local_path, trust_remote_code=trust, use_fast=True)
        envs, val_envs = make_envs(config)
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        trainer = FrontierOPSDTrainer(config=config, tokenizer=tokenizer, processor=processor,
            worker_cls=ray.remote(ActorRolloutRefWorker), train_dataset=train_dataset,
            val_dataset=val_dataset, collate_fn=collate_fn,
            train_sampler=create_rl_sampler(config.data, train_dataset),
            traj_collector=FrontierTrajectoryCollector(config, tokenizer, processor),
            envs=envs, val_envs=val_envs,
            val_reward_fn=EpisodeRewardManager(tokenizer=tokenizer, num_examine=1, normalize_by_length=False),
            device_name=config.trainer.device)
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
