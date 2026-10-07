# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Nanyang Technological University (NTU) and the verl-agent team.
# Adapted from SDAR/veRL under the Apache License, Version 2.0.
# See LICENSE in the repository root.
"""Dataset loading and SDAR validation utilities, without an RL trainer.

Validation methods are retained from SDAR to preserve its evaluation protocol.
"""

import numpy as np
import torch
from omegaconf import open_dict
from torchdata.stateful_dataloader import StatefulDataLoader
from verl import DataProto
from verl.utils.tracking import ValidationGenerationsLogger


class FrontierRuntime:
    def __init__(self, config, tokenizer, worker_cls, train_dataset, val_dataset,
                 collate_fn, train_sampler, traj_collector, envs, val_envs,
                 val_reward_fn, processor=None, device_name="cuda"):
        self.config, self.tokenizer, self.processor = config, tokenizer, processor
        self.worker_cls, self.device_name = worker_cls, device_name
        self.traj_collector, self.envs, self.val_envs = traj_collector, envs, val_envs
        self.val_reward_fn = val_reward_fn
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.train_dataloader = StatefulDataLoader(
            train_dataset, batch_size=config.data.get("gen_batch_size", config.data.train_batch_size),
            num_workers=config.data.get("dataloader_num_workers", 8), drop_last=False,
            collate_fn=collate_fn, sampler=train_sampler)
        self.val_dataloader = StatefulDataLoader(
            val_dataset, batch_size=config.data.get("val_batch_size") or len(val_dataset),
            num_workers=config.data.get("dataloader_num_workers", 8), drop_last=False,
            shuffle=False, collate_fn=collate_fn)
        if not len(self.train_dataloader) or not len(self.val_dataloader):
            raise ValueError("Frontier training and validation datasets must be nonempty")
        self.total_training_steps = config.trainer.get("total_training_steps")
        if self.total_training_steps is None:
            self.total_training_steps = len(self.train_dataloader) * config.trainer.total_epochs
        with open_dict(config):
            config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps

    def init_workers(self):
        from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
        from verl.single_controller.ray.base import create_colocated_worker_cls
        self.resource_pool = RayResourcePool(
            [self.config.trainer.n_gpus_per_node] * self.config.trainer.nnodes,
            use_gpu=True, max_colocate_count=1, name_prefix="frontier")
        actor_cls = RayClassWithInitArgs(cls=self.worker_cls,
            config=self.config.actor_rollout_ref, role="actor_rollout")
        colocated = create_colocated_worker_cls({"actor_rollout": actor_cls})
        kwargs = {}
        timeout = self.config.trainer.get("ray_wait_register_center_timeout")
        if timeout is not None:
            kwargs["ray_wait_register_center_timeout"] = timeout
        # Keep the parent group alive for the lifetime of its spawned workers.
        self.worker_group = RayWorkerGroup(resource_pool=self.resource_pool,
            ray_cls_with_init=colocated, device_name=self.device_name, **kwargs)
        self.actor_rollout_wg = self.worker_group.spawn(prefix_set={"actor_rollout"})["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _maybe_log_val_generations(self, inputs, outputs, scores):
            """Log a table of validation samples to the configured logger (wandb or swanlab)"""
    
            generations_to_log = self.config.trainer.log_val_generations
    
            if generations_to_log == 0:
                return
    
            import numpy as np
    
            # Create tuples of (input, output, score) and sort by input text
            samples = list(zip(inputs, outputs, scores))
            samples.sort(key=lambda x: x[0])  # Sort by input text
    
            # Use fixed random seed for deterministic shuffling
            rng = np.random.RandomState(42)
            rng.shuffle(samples)
    
            # Take first N samples after shuffling
            samples = samples[:generations_to_log]
    
            # Log to each configured logger
            self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
            reward_tensor_lst = []
            data_source_lst = []
            tool_calling_list = []
            traj_uid_list = []
            success_rate_dict = {}
    
            # Lists to collect samples for the table
            sample_inputs = []
            sample_outputs = []
            sample_scores = []
    
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)
    
                # repeat test batch
                test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)
    
                # Store original inputs
                input_ids = test_batch.batch["input_ids"]
                # TODO: Can we keep special tokens except for padding tokens?
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)
    
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                test_gen_batch = test_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
    
                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                }
                print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")
    
                # # pad to be divisible by dp_size
                # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
    
                # # unpad
                # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
    
                ################ agent-environment loop ###############
                test_output_gen_batch = self.traj_collector.multi_turn_loop(
                                                        gen_batch=test_gen_batch,
                                                        actor_rollout_wg=self.actor_rollout_wg,
                                                        envs=self.val_envs,
                                                        is_train=False,
                                                        )
                print('validation generation end')
                del test_batch
                test_batch = test_output_gen_batch
                # Store generated outputs
                output_ids = test_output_gen_batch.batch["responses"]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)
    
                # test_batch = test_batch.union(test_output_gen_batch)
    
                # evaluate using reward_function
                result = self.val_reward_fn(test_batch, return_dict=True)
                reward_tensor = result["reward_tensor"]
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)
    
                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
                tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
                traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
                # success rate
                for k in test_batch.non_tensor_batch.keys():
                    if 'success_rate' in k:
                        if k not in success_rate_dict:
                            success_rate_dict[k] = []
                        success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                        # all success_rate should be the same
                        for i in range(1, len(test_batch.non_tensor_batch[k])):
                            assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'
    
            self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
    
            reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
            data_sources = np.concatenate(data_source_lst, axis=0)
            tool_callings = np.concatenate(tool_calling_list, axis=0)
            traj_uids = np.concatenate(traj_uid_list, axis=0)
            success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}
    
            # evaluate test_score based on data source
            data_source_reward = {}
            for i in range(reward_tensor.shape[0]):
                data_source = data_sources[i]
                if data_source not in data_source_reward:
                    data_source_reward[data_source] = []
                data_source_reward[data_source].append(reward_tensor[i].item())
    
            # evaluate tool call based on data source
            # the values in tool_callings represent the tool call count for each trajectory; however, since the batch is expanded by step, we only need to take one value for each unique trajectories.
            data_source_tool_calling = {}
            unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
            unique_data_sources = data_sources[unique_idx]
            unique_tool_callings = tool_callings[unique_idx]
    
            for i in range(unique_tool_callings.shape[0]):
                data_source = unique_data_sources[i]
                if data_source not in data_source_tool_calling:
                    data_source_tool_calling[data_source] = []
                data_source_tool_calling[data_source].append(unique_tool_callings[i].item())
    
            metric_dict = {}
            for data_source, rewards in data_source_reward.items():
                metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)
    
            for data_source, tool_calls in data_source_tool_calling.items():
                metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
                # metric_dict[f'val/{data_source}/tool_call_count/max'] = np.max(tool_calls)
                # metric_dict[f'val/{data_source}/tool_call_count/min'] = np.min(tool_calls)
    
            for k, v in success_rate.items():
                metric_dict[f'val/{k}'] = v
    
            return metric_dict
