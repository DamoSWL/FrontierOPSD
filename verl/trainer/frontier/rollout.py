"""Method-only rollout extensions over SDAR's existing environments and prompts."""

import json
from copy import deepcopy
from frontier_opsd.records import task_id
import numpy as np
import torch

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector, _gamefiles_from_infos
from agent_system.multi_turn_rollout.utils import to_list_of_dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def generate(actor, prompts):
    padded, padding = pad_dataproto_to_divisor(prompts, actor.world_size)
    return unpad_dataproto(actor.generate_sequences(padded), padding)


class FrontierTrajectoryCollector(TrajectoryCollector):
    """Collect unhinted on-policy trajectories and score original frontier tokens."""

    def scoring_target(self, original, hint):
        from verl.utils.model import compute_position_id_with_mask

        row = deepcopy(original)
        length = row["responses"].shape[-1]
        row["response_mask"] = row["attention_mask"][-length:].clone()
        prompt = row["input_ids"][:-length]
        valid = row["attention_mask"][:-length].bool()
        hint_ids = torch.tensor(self.tokenizer.encode(
            "[Privileged diagnostic hint]\n" + hint + "\n\n",
            add_special_tokens=False), dtype=prompt.dtype, device=prompt.device)
        # Preserve the original prompt IDs and response IDs exactly. Never
        # truncate away the hint or the pre-action history silently.
        teacher_prompt = torch.cat((hint_ids, prompt[valid]))
        if teacher_prompt.numel() > self.config.data.max_prompt_length:
            raise ValueError("Hint-augmented frontier prompt exceeds data.max_prompt_length")
        row["teacher_input_ids"] = torch.cat((teacher_prompt, row["responses"]))
        row["teacher_attention_mask"] = torch.cat((torch.ones_like(teacher_prompt), row["response_mask"]))
        row["teacher_position_ids"] = compute_position_id_with_mask(
            row["teacher_attention_mask"].unsqueeze(0))[0]
        return row

    def collect(self, gen_batch, actor, envs, repeat_tasks=False, task_specs=None, task_count=None, excluded_task_ids=None):
        size = len(gen_batch)
        reset_kwargs = deepcopy(gen_batch.non_tensor_batch.get("env_kwargs"))
        environment = self.config.env.env_name.lower()
        if "search" not in environment:
            reset_kwargs = {"frontier_task_ids": [spec["identity"] for spec in task_specs]
                            if task_specs else "next", "frontier_task_count": task_count or size, "frontier_excluded_task_ids": excluded_task_ids or []}
        obs, infos = envs.reset(kwargs=reset_kwargs)
        gamefiles = _gamefiles_from_infos(infos)
        episodes = [{"steps": [], "infos": [], "success": False} for _ in range(size)]
        for i, episode in enumerate(episodes):
            if "alfworld" in environment:
                identity = str(gamefiles[i])
            elif "webshop" in environment:
                identity = int(infos[i]["frontier_session_id"])
            else:
                # Search task identity follows question/ground-truth dataset
                # kwargs, not dataloader order or mutable episode observations.
                identity = reset_kwargs[i] if reset_kwargs is not None else obs["text"][i]
                if hasattr(identity, "tolist"):
                    identity = identity.tolist()
            identity = json.loads(json.dumps(identity, default=lambda value: value.tolist()
                if hasattr(value, "tolist") else str(value)))
            episode["task_spec"] = {"environment": environment, "identity": identity}
            episode["task_id"] = task_id(environment, identity)
        done = np.zeros(size, dtype=bool)
        if task_count is not None:
            done[task_count:] = True
        rewards_total = np.zeros(size)
        lengths = np.zeros(size)
        for step in range(self.config.env.max_steps):
            active = ~done
            before = list(obs["text"])
            plain = self.preprocess_batch(gen_batch, {**obs, "gamefile": gamefiles})
            keys = ["input_ids", "attention_mask", "position_ids"]
            non_keys = [key for key in ("raw_prompt_ids", "raw_prompt", "multi_modal_data")
                        if key in plain.non_tensor_batch]
            prompts = plain.select(keys, non_keys)
            prompts.meta_info = deepcopy(gen_batch.meta_info)
            output = generate(actor, prompts)
            actions = self.tokenizer.batch_decode(output.batch["responses"], skip_special_tokens=True)
            next_obs, rewards, dones, infos = envs.step(actions)
            rewards = np.asarray(rewards).reshape(-1)
            dones = np.asarray(dones).reshape(-1)
            rewards_total[active] += rewards[active]
            lengths[active] += 1

            # Preserve the original sampled response and EOS/padding mask.
            response_length = output.batch["responses"].shape[-1]
            response_mask = output.batch["attention_mask"][:, -response_length:]
            from verl.utils.model import compute_position_id_with_mask
            student_ids = torch.cat((plain.batch["input_ids"], output.batch["responses"]), -1)
            student_mask = torch.cat((plain.batch["attention_mask"], response_mask), -1)
            output.batch["input_ids"] = student_ids
            output.batch["attention_mask"] = student_mask
            output.batch["position_ids"] = compute_position_id_with_mask(student_mask)
            if "prompts" in output.batch:
                output.batch["prompts"] = plain.batch["input_ids"]
            # Preserve the original student prompt metadata.
            for key in list(output.non_tensor_batch):
                if key in plain.non_tensor_batch:
                    output.non_tensor_batch.pop(key)
            plain.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
            batch = plain.union(output)
            batch.non_tensor_batch["active_masks"] = active.astype(object)
            rows = to_list_of_dict(batch)
            for i, row in enumerate(rows):
                if not active[i]:
                    continue
                row.update(frontier_before=before[i], frontier_after=next_obs["text"][i],
                           frontier_action=actions[i], turn_step=step, rewards=float(rewards[i]),
                           frontier_target=False)
                episodes[i]["steps"].append(row)
                episodes[i]["infos"].append(infos[i])
            done |= dones
            obs = next_obs
            gamefiles = _gamefiles_from_infos(infos)
            if done.all():
                break
        real_count = task_count or size
        success = envs.success_evaluator(
            total_infos=[episode["infos"] for episode in episodes[:real_count]],
            total_batch_list=[episode["steps"] for episode in episodes[:real_count]],
            episode_rewards=rewards_total[:real_count], episode_lengths=lengths[:real_count])
        for i, episode in enumerate(episodes[:real_count]):
            episode["success"] = bool(success["success_rate"][i])
        return episodes
