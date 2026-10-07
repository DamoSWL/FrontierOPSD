import unittest

from frontier_opsd.adapters import StructuredCritic
from frontier_opsd.demo import DemoCritic, DemoEnvironment, DemoPolicy
from frontier_opsd.trainer import Diagnosis, FrontierConfig, FrontierTrainer, History


class FrontierTests(unittest.TestCase):
    def test_complete_epoch_before_revisiting_and_success_is_revisited(self):
        visits = []
        class RecordingEnvironment(DemoEnvironment):
            def reset(self, task):
                visits.append(task)
                return super().reset(task)
        policy = DemoPolicy()
        trainer = FrontierTrainer(policy, RecordingEnvironment(), DemoCritic(), FrontierConfig(total_epochs=4))
        results = list(trainer.fit(["taskA", "taskB"]))
        self.assertEqual(visits, ["taskA", "taskB"] * 4)
        self.assertEqual(len(results), 8)
        self.assertEqual(trainer.policy_version, 3)
        self.assertEqual(results[-1]["status"], "success")
        self.assertEqual(results[0]["trajectory"].history.actions, ("wrong",))
        self.assertEqual(results[0]["events"][0]["epoch"], 1)

    def test_one_rollout_and_one_diagnosis_no_post_update_rollout(self):
        class CountingEnvironment(DemoEnvironment):
            calls = 0
            def reset(self, task):
                self.calls += 1
                return super().reset(task)
            def restore(self, history):
                raise AssertionError("No verification")
        class CountingCritic(DemoCritic):
            calls = 0
            def diagnose(self, trajectory):
                self.calls += 1
                return super().diagnose(trajectory)
        env, critic = CountingEnvironment(), CountingCritic()
        trainer = FrontierTrainer(DemoPolicy(), env, critic)
        result = trainer.train_task("task")
        self.assertEqual(result["status"], "updated")
        self.assertEqual((env.calls, critic.calls), (1, 1))
        self.assertEqual(result["trajectory"].history.actions, ("wrong",))

    def test_stale_teacher_target_cannot_trigger_update(self):
        class StaleTeacher(DemoPolicy):
            def target(self, history, action, hint, version):
                from frontier_opsd.trainer import Target
                return Target(history, "advance:0", (), version - 1)

        class StalePolicy(DemoPolicy):
            def freeze(self):
                return StaleTeacher()

        trainer = FrontierTrainer(StalePolicy(), DemoEnvironment(), DemoCritic())
        with self.assertRaisesRegex(ValueError, "policy version"):
            trainer.train_task("task")
        self.assertEqual(trainer.policy_version, 0)

    def test_success_does_not_call_critic(self):
        class NoCritic:
            def diagnose(self, trajectory):
                raise AssertionError("No diagnosis needed")

        trainer = FrontierTrainer(DemoPolicy([0, 1, 2]), DemoEnvironment(), NoCritic())
        self.assertEqual(trainer.train_task("task")["status"], "success")
        self.assertFalse(hasattr(trainer, "memory"))

    def test_invalid_diagnosis_fails_closed(self):
        for diagnosis in (Diagnosis(-1, "hint"), Diagnosis(1, "hint"), Diagnosis(True, "hint"), Diagnosis(0, "")):
            class InvalidCritic:
                def diagnose(self, trajectory):
                    return diagnosis
            with self.subTest(diagnosis=diagnosis):
                trainer = FrontierTrainer(DemoPolicy(), DemoEnvironment(), InvalidCritic())
                with self.assertRaises(ValueError):
                    trainer.train_task("task")

    def test_config_and_history_validation(self):
        for kwargs in ({"total_epochs": 0}, {"max_steps": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FrontierConfig(**kwargs)
        with self.assertRaises(ValueError):
            History("task", (), ())

    def test_critic_rejects_external_action_target(self):
        critic = StructuredCritic(lambda *_: {"frontier": 0, "hint": "hint", "action": "click"})
        with self.assertRaises(ValueError):
            critic.diagnose(None)


if __name__ == "__main__":
    unittest.main()
