"""Dependency-free example of fresh current-policy progressive distillation."""

from .trainer import Diagnosis, Target


class DemoEnvironment:
    def reset(self, task):
        self.position = 0
        self.failed = False
        return self.position

    def step(self, action):
        if action != f"advance:{self.position}":
            self.failed = True
        else:
            self.position += 1
        return self.position, self.failed or self.position == 3

    def succeeded(self):
        return self.position == 3 and not self.failed

    def restore(self, history):
        self.position = history.observations[-1]
        self.failed = False
        return self.position


class DemoCritic:
    def diagnose(self, trajectory):
        for index, action in enumerate(trajectory.history.actions):
            if action != f"advance:{trajectory.history.observations[index]}":
                return Diagnosis(index, "Advance from the current position")
        return None


class DemoPolicy:
    def __init__(self, learned=()):
        self.learned = set(learned)

    def act(self, history):
        position = history.observations[-1]
        return f"advance:{position}" if position in self.learned else "wrong"

    def freeze(self):
        return DemoPolicy(self.learned)

    def target(self, history, action, hint, version):
        return Target(history, action, (action,), version)

    def update(self, target):
        self.learned.add(target.history.observations[-1])
        return {}


def build(config):
    return DemoPolicy(), DemoEnvironment(), DemoCritic(), ["reach position three"]
