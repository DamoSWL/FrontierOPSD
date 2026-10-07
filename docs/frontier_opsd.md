# FrontierOPSD implementation

The current method uses SDAR's ALFWorld, WebShop, and Search-QA datasets,
original prompts, and validation protocol. The reference SDAR checkout is unchanged.

## Training loop

1. Collect fresh unhinted current-policy trajectories.
2. Ask an independent stronger critic for `{frontier, hint}`. The frontier is a
   zero-based action index in the original trajectory, which includes the mistake.
3. Select exactly that original step. Preserve its response IDs and EOS mask.
4. Cache unhinted old-policy token scores and hint-conditioned frozen self-teacher
   scores BEFORE optimization. Both score the same tokens with the same preceding
   response prefix; the teacher additionally receives the hint.
5. Call the original `compute_sdl_loss` from `verl/trainer/ppo/skillsd_utils.py`.
   It computes sampled K3-IS OPSD, with its existing exponent caps unchanged.
   Normalize each step by valid tokens and average steps equally. Distributed
   padding examples carry zero weight.
6. Discard cached scores and move to the next batch. After a complete pass,
   revisit each task with one new unhinted rollout in the next epoch. Successful
   rollouts skip training for that epoch but remain in later epochs.

No repaired action is generated. No prefix restoration, candidate rollout,
hint verification, full-vocabulary forward KL, PPO clipping, GRPO loss, advantage
estimator, or historical replay is used. The importance ratio in OPSD is retained.
The standalone Torch backend caches actual rollout IDs rather than re-tokenizing
an action string. Its frozen model snapshot provides both scoring contexts.

## Configuration and execution

Run from the FrontierOPSD root with the normal SDAR dependencies installed:

```bash
bash examples/frontier_opsd/run_sdar_dataset.sh alfworld vllm \
    +algorithm.frontier.objective=sampled_opsd \
    +algorithm.frontier.critic_model=Qwen/Qwen2.5-32B-Instruct \
    trainer.logger='[console]'
```

Use `webshop` or `search` for the other existing launch configurations.
`objective=sampled_opsd` is the default and only supported objective.
`update_steps=1` is the default. `trainer.total_epochs` controls data passes.
Per-batch `max_rounds` settings are rejected, as is `env.rollout.n != 1`. Repair `candidates` settings are rejected.
The separate critic supports OpenAI Responses and compatible chat-completions
servers; see the README for endpoint configuration. Credentials remain outside
Hydra config. Invalid diagnoses are skipped; endpoint failures raise errors.

The dedicated trainer inherits only FrontierRuntime and never calls a PPO/GRPO
trainer. Scoring holds actor weights fixed, so distributed training does not need
another student worker. Only sampled token scores are cached on CPU.
Hint IDs are prepended to the exact original valid prompt IDs. Prompts exceeding
`data.max_prompt_length` raise rather than silently truncating history or hints.

## Implementation and validation

- `verl/trainer/frontier/rollout.py`: unhinted collection and original-step targets.
- `verl/trainer/frontier/trainer.py`: diagnosis, frozen score caching, current-round
  updates, fresh evaluation rollouts, validation, and checkpoints.
- `verl/workers/actor/dp_actor.py`: sampled scoring and existing OPSD loss reuse.
- `frontier_opsd/`: standalone reference loop, Torch backend, and toy demo.

```bash
python -m unittest discover -s tests/frontier_opsd -v
python -m frontier_opsd examples/frontier_opsd/demo.json
```

Training supports synchronous vLLM, text inputs, FSDP/FSDP2, and sequence
parallelism size 1. Checkpoints preserve optimizer, data-loader, and policy-version
state, not in-progress trajectories or cached teacher scores. A resumed run starts
a fresh rollout. Full multi-GPU environment training and benchmark results remain
to be validated. The demo is a control-flow illustration, not a benchmark.

## Epoch history

The trainer writes `trainer.default_local_dir/rollouts.jsonl` with stable task ID,
epoch, rollout-time policy version, success, original response tokens and indexed
observations/actions, diagnosis/hint, and a comparison reason. Successful rollouts
are recorded. Comparisons are noisy single-rollout estimates: outcome changes
are primary; aligned non-repeating prefix depth gives supporting evidence.
Different paths, repetition, or missing diagnoses yield `uncertain`. Historical
records never supply training tokens, including after regressions.

ALFWorld/WebShop use unique first-epoch scheduled gamefiles/session IDs and reset
those same tasks in subsequent epochs. The driver budget must fit the environment
split. Search IDs are derived from dataset task kwargs. Final partial batches are
padded for fixed-size environment workers, with padding excluded from logs/loss.
Checkpoints include epoch/batch position, task schedules, and latest comparison
records. The standalone demo uses `total_epochs` and processes all tasks per pass.

## Open-source critic with vLLM

The default `critic_api=vllm` uses `vllm.LLM.generate()` directly in an
application-managed Ray GPU worker, without an HTTP server. The model is loaded
once. Its tokenizer formats the indexed text trace with the chat template.
Qwen3 thinking is disabled by default; complete reasoning blocks are removed
before validating the JSON diagnosis.

Set `critic_model=Qwen/Qwen3-32B` or a local checkpoint path. The critic reserves
`critic_tensor_parallel_size` additional GPUs (default 2) beyond the student's
`trainer.n_gpus_per_node * trainer.nnodes` allocation. Expose all these GPUs to
Ray and install vLLM in the training environment. Defaults are
`critic_gpu_memory_utilization=0.85`, `critic_max_model_len=16384`, and
`critic_trust_remote_code=false`. Context overflows raise an explicit error;
truncated or malformed answers are skipped. Hosted `responses` and generic
`chat_completions` backends remain supported.
