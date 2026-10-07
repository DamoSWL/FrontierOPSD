# FrontierOPSD

**Frontier-Guided On-Policy Self-Distillation for Multi-Turn Agents**

FrontierOPSD trains an agent at the earliest consequential mistake in its current trajectory. A stronger critic identifies the mistaken step and supplies a hint. A frozen current student re-scores the original step tokens with the hint; the trainable student learns without seeing it.

The implementation reuses [SDAR](https://github.com/ZJU-REAL/SDAR)'s ALFWorld, WebShop, and Search-QA datasets, prompts, task splits, and validation protocol.

![FrontierOPSD overview](FrontierOPSD.png)

## Method

Training proceeds in complete passes over the configured training tasks. Each task gets one fresh, unhinted rollout per epoch, using the student weights at the time its batch is visited.

1. Collect a fresh trajectory from the current student without hints. Record task success using the environment outcome.
2. Skip successful trajectories for training. For a failed trajectory, ask the separate stronger critic for the earliest consequential error index and a hint.
3. Keep the original mistaken-step response and pre-action context.
4. Before any update, score those same tokens with the frozen current student twice: with the hint for teacher scores, and without it for rollout-policy scores.
5. Minimize KL divergence between the student and the hint-conditioned self-teacher at the mistaken step. The student receives the original context without the hint.
6. Move to the next batch. Only after the full data pass, start the next epoch and generate fresh unhinted trajectories. Successful tasks skip training for this epoch and are still revisited next epoch.



## KL distillation objective

At each batch, let $\theta_{\mathrm{old}}$ denote the student weights used to generate the fresh, unhinted trajectories. The critic identifies a mistaken step and supplies a hint $h$. At each token position $t$ of that step, define the student and frozen self-teacher distributions:

$$
p_\theta^t(y)=\pi_{student}(y_t\mid c,y_{<t}),
\qquad
q^t(y)=\pi_{{teacher}}(y_t\mid c,h,y_{<t}).
$$

Here, $c$ is the original pre-action context, and $y_{<t}$ contains the preceding tokens from the original student response. Both models score the same original response; the hint is added only to the teacher context.

The objective is reverse KL divergence, averaged over diagnosed steps and their valid token positions:

$$
\boxed{
\mathcal L_{\mathrm{KL}}(\theta)
=
\frac{1}{|\mathcal B|}\sum_{i\in\mathcal B}
\frac{1}{|M_i|}\sum_{t\in M_i}
D_{\mathrm{KL}}\!\left(p_\theta^{i,t}\,\Vert\,q^{i,t}\right)
}
$$

where $\mathcal B$ contains the current batch's diagnosed failures and $M_i$ selects the valid tokens of the original mistaken step. At a given prefix,

$$
D_{\mathrm{KL}}(p\Vert q)=\sum_{y\in Y}p(y)\log\frac{p(y)}{q(y)}.
$$

### Sampled implementation

The implementation uses the existing OPSD sampled estimator rather than explicitly summing over the vocabulary. For each original rollout token $y_t$, define

$$
\ell_t=\log p_\theta^t(y_t)-\log q^t(y_t),
\qquad
\rho_t=\frac{p_\theta^t(y_t)}{p_{\mathrm{old}}^t(y_t)}.
$$

The estimated loss averages $\rho_t(e^{-\ell_t}-1+\ell_t)$ over the valid mistaken-step tokens and then over examples. This is the reference K3 estimator with importance weighting; its numerical exponent caps are retained. It is a sampled approximation to the KL objective above.

Teacher and unhinted rollout-policy scores are cached before optimization and held fixed during the update. The distributed implementation uses the current student for both scoring passes while its weights remain frozen. Cached scores are discarded after the batch update. The stronger critic supplies text hints; its logits are not required.


## Installation and data preparation

Run the following commands from the **FrontierOPSD repository root**. Training currently supports text environments, synchronous vLLM, FSDP/FSDP2, and sequence parallelism size 1.

### ALFWorld and Search training environment

The following package versions are inherited from SDAR's installation instructions:

```bash
conda create -n frontieropsd python=3.12 -y
conda activate frontieropsd
pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

For ALFWorld, install the environment and download its games:

```bash
pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
alfworld-download -f
export ALFWORLD_DATA="$HOME/data/alfworld"
```

The ALFWorld and WebShop launch scripts generate SDAR's driver parquet files at `~/data/verl-agent/text/train.parquet` and `~/data/verl-agent/text/test.parquet`.

### WebShop environment

WebShop follows SDAR's separate Python 3.10 environment:

```bash
conda create -n frontieropsd-webshop python=3.10 -y
conda activate frontieropsd-webshop
(
    cd agent_system/environments/env_package/webshop/webshop
    bash setup.sh -d all
)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e .
pip install vllm==0.8.2
```

## Run FrontierOPSD


### ALFWorld

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
bash examples/frontier_opsd/run_sdar_dataset.sh alfworld vllm \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    +algorithm.frontier.critic_api=vllm \
    +algorithm.frontier.critic_model=Qwen/Qwen3-32B \
    +algorithm.frontier.critic_tensor_parallel_size=2 \
    +actor_rollout_ref.actor.frontier_micro_batch_size_per_gpu=1 \
    trainer.experiment_name=frontieropsd_alfworld_qwen2.5_3b \
    trainer.logger='[console]'
```

### WebShop

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
bash examples/frontier_opsd/run_sdar_dataset.sh webshop vllm \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    +algorithm.frontier.critic_api=vllm \
    +algorithm.frontier.critic_model=Qwen/Qwen3-32B \
    +algorithm.frontier.critic_tensor_parallel_size=2 \
    +actor_rollout_ref.actor.frontier_micro_batch_size_per_gpu=1 \
    trainer.experiment_name=frontieropsd_webshop_qwen2.5_3b \
    trainer.logger='[console]'
```


## Acknowledgments

FrontierOPSD is adapted from [SDAR](https://github.com/ZJU-REAL/SDAR) and reuses infrastructure from [veRL](https://github.com/volcengine/verl) and [verl-agent](https://github.com/langfengQ/verl-agent), with environments from [ALFWorld](https://github.com/alfworld/alfworld), [WebShop](https://github.com/princeton-nlp/WebShop), and [Search-R1](https://github.com/PeterGriffinJin/Search-R1). The repository retains the upstream [Apache 2.0 license](LICENSE) and attribution.
