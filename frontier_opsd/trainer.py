"""Rollout → diagnose → re-score original tokens with hint → distill → fresh rollout.

The critic identifies the earliest consequential error and supplies a hint only.
The behavioral teacher is always a frozen copy of the current student. Backend
interfaces deliberately carry complete environment observations intact.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol
from .records import RolloutRecords, task_id


@dataclass(frozen=True)
class History:
    task: Any
    observations: tuple[Any, ...]
    actions: tuple[Any, ...]

    def __post_init__(self):
        if len(self.observations) != len(self.actions) + 1:
            raise ValueError("History must contain one observation before each action plus the current observation")


@dataclass(frozen=True)
class Trajectory:
    history: History
    success: bool


@dataclass(frozen=True)
class Diagnosis:
    # Zero-based index of the action that first causes consequential failure.
    frontier: int
    hint: str


@dataclass(frozen=True)
class Target:
    # Backend-owned immutable payload: sampled tokens and/or cached teacher logits.
    history: History
    action: Any
    payload: Any
    policy_version: int


class Environment(Protocol):
    def reset(self, task: Any) -> Any: ...
    def step(self, action: Any) -> tuple[Any, bool]: ...
    def succeeded(self) -> bool: ...


class Critic(Protocol):
    def diagnose(self, trajectory: Trajectory) -> Diagnosis | None: ...


class FrozenPolicy(Protocol):
    def target(self, history: History, action: Any, hint: str, version: int) -> Target:
        """Re-score the original action with the hint before updates."""
        ...
    def act(self, history: History) -> Any: ...


class Policy(Protocol):
    def act(self, history: History) -> Any: ...
    def freeze(self) -> FrozenPolicy: ...
    def update(self, target: Target) -> dict:
        """Minimize KL on this round's original current-policy frontier."""
        ...


@dataclass(frozen=True)
class FrontierConfig:
    max_steps: int = 30
    total_epochs: int = 10
    record_file: str | None = None

    def __post_init__(self):
        for name in ("max_steps", "total_epochs"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class FrontierTrainer:
    def __init__(self, policy: Policy, env: Environment, critic: Critic, config=None):
        self.policy, self.env, self.critic = policy, env, critic
        self.config = config or FrontierConfig()
        self.policy_version = 0
        self.records = RolloutRecords(self.config.record_file)

    def _continue(self, history: History, policy, first_action=None) -> Trajectory:
        observations, actions = list(history.observations), list(history.actions)
        while len(actions) < self.config.max_steps:
            current = History(history.task, tuple(observations), tuple(actions))
            action = first_action if first_action is not None else policy.act(current)
            first_action = None
            observation, done = self.env.step(action)
            actions.append(deepcopy(action))
            observations.append(deepcopy(observation))
            if done:
                break
        return Trajectory(History(history.task, tuple(observations), tuple(actions)), bool(self.env.succeeded()))

    def rollout(self, task) -> Trajectory:
        if hasattr(self.policy, "begin_rollout"):
            self.policy.begin_rollout()
        observation = deepcopy(self.env.reset(task))
        return self._continue(History(deepcopy(task), (observation,), ()), self.policy)

    def _diagnose(self, trajectory):
        if trajectory.success:
            return None
        diagnosis = self.critic.diagnose(deepcopy(trajectory))
        if diagnosis is None:
            return None
        if type(diagnosis.frontier) is not int or not 0 <= diagnosis.frontier < len(trajectory.history.actions):
            raise ValueError("Critic frontier must be a zero-based action index in this trajectory")
        if not isinstance(diagnosis.hint, str) or not diagnosis.hint.strip():
            raise ValueError("Critic must supply a nonempty repair hint")
        return diagnosis

    def train_task(self, task, epoch=1) -> dict:
        # Exactly one fresh rollout. Even a successful task returns next epoch.
        trajectory = self.rollout(task)
        diagnosis = self._diagnose(trajectory)
        record = {"task_id": task_id("standalone", task), "epoch": epoch,
            "policy_version": self.policy_version, "success": trajectory.success,
            "task": str(task), "steps": [{"step_index": i,
                "observation_before": str(trajectory.history.observations[i]),
                "response": str(action), "action": str(action),
                "observation_after": str(trajectory.history.observations[i + 1])}
                for i, action in enumerate(trajectory.history.actions)],
            "diagnosis": None if diagnosis is None else {
                "mistaken_step_index": diagnosis.frontier, "hint": diagnosis.hint},
            "progress": {}}
        self.records.add(record)
        if trajectory.success or diagnosis is None:
            return {"status": "success" if trajectory.success else "undiagnosed",
                    "events": [], "trajectory": trajectory, "record": record}
        t = diagnosis.frontier
        history = History(deepcopy(task), trajectory.history.observations[:t + 1], trajectory.history.actions[:t])
        teacher = self.policy.freeze()
        teacher_history = deepcopy(history)
        original_action = deepcopy(trajectory.history.actions[t])
        target = teacher.target(teacher_history, original_action, diagnosis.hint, self.policy_version)
        if target.history is not teacher_history or target.policy_version != self.policy_version:
            raise ValueError("Teacher target must preserve the unprivileged history and policy version")
        if target.action != original_action:
            raise ValueError("Teacher must re-score the original rollout action")
        del teacher
        event = {"epoch": epoch, "policy_version": self.policy_version, "frontier": t}
        event.update(self.policy.update(target))
        self.policy_version += 1
        return {"status": "updated", "events": [event], "trajectory": trajectory, "record": record}

    def fit(self, tasks):
        tasks = list(tasks)
        if len({task_id("standalone", task) for task in tasks}) != len(tasks):
            raise ValueError("Training task IDs must be unique within an epoch")
        for epoch in range(1, self.config.total_epochs + 1):
            for task in tasks:
                yield self.train_task(task, epoch)
