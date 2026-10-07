"""Progressive frontier on-policy self-distillation, independent of PPO."""

from .trainer import Diagnosis, FrontierConfig, FrontierTrainer, History, Target, Trajectory

__all__ = ["Diagnosis", "FrontierConfig", "FrontierTrainer", "History", "Target", "Trajectory"]
