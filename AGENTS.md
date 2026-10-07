# FrontierOPSD project scope

Use SDAR's ALFWorld, WebShop, and Search-QA datasets and environments.
Do not introduce GUI datasets, OSWorld, or GUI-Odyssey.

The agreed method is fresh unprivileged student rollout → earliest consequential
error diagnosis and hint → frozen current-student re-scoring of the ORIGINAL
mistaken-step tokens with the hint → sampled OPSD update without the hint →
next epoch's fresh rollout after a full pass over all task batches.
Use exactly one rollout per task per epoch. Skip successful rollouts for training
but revisit them next epoch. No immediate post-update training rollout.
Keep stable task identities and JSONL epoch records for comparison only;
never train on historical trajectories even when fresh performance regresses. Do not generate repair targets or verify hints.
Preserve original response IDs and EOS masks. Score each token with only its
preceding response prefix, never the complete response as an extra prompt.
Every training anchor and sampled token must come from the current rollout.
No historical replay buffer or replay loss.

Use a separate stronger critic for localization and hints, supporting hosted
OpenAI and open-source models using the in-process vLLM Python API. The student remains the
self-teacher; no external critic logits are needed.
Reuse verl/trainer/ppo/skillsd_utils.py:compute_sdl_loss unchanged: sampled K3
with importance weighting against cached unhinted rollout-policy scores.
OPSD is the sole training loss. Do not add PPO/GRPO clipping, advantages,
reward-based policy gradients, or full-vocabulary forward KL. The dedicated
trainer must not inherit the PPO/GRPO pipeline. Keep teacher and old-policy
scores frozen for the current update, then discard them.

See docs/frontier_opsd.md for implementation and validation limits.
