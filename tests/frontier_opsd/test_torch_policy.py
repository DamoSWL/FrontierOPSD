import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class TorchTests(unittest.TestCase):
    def test_sampled_opsd_matches_reference_and_freezes_scores(self):
        from frontier_opsd.losses import distillation_loss
        from verl.trainer.ppo.skillsd_utils import compute_sdl_loss
        student = torch.tensor([-1.3, -2.1], requires_grad=True)
        teacher = torch.tensor([-.4, -3.], requires_grad=True)
        old = torch.tensor([-1.5, -2.], requires_grad=True)
        loss = distillation_loss(student, teacher, old, torch.ones(2))
        expected = compute_sdl_loss(student.unsqueeze(0), teacher.unsqueeze(0),
                                    old.unsqueeze(0), torch.ones(1, 2))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(old.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_identical_scores_and_padding(self):
        from frontier_opsd.losses import distillation_loss
        student = torch.tensor([-1., -2.], requires_grad=True)
        teacher = torch.tensor([-1., float("nan")])
        loss = distillation_loss(student, teacher, student.detach(), torch.tensor([1., 0.]))
        torch.testing.assert_close(loss, torch.tensor(0.))
        loss.backward()
        torch.testing.assert_close(student.grad, torch.zeros(2))

    def test_model_snapshot_hint_isolation_and_current_round_update(self):
        from types import SimpleNamespace
        from frontier_opsd.torch_policy import TorchPolicy
        from frontier_opsd.trainer import History

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.table = torch.nn.Parameter(torch.tensor([[2., 0., 0.], [0., 0., 2.], [0., 0., 0.]]))
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.table[input_ids])
            def generate(self, input_ids, **kwargs):
                return torch.cat((input_ids, torch.tensor([[2]])), -1)

        encoded_hints = []
        def encode(history, hint):
            encoded_hints.append(hint)
            return {"input_ids": torch.tensor([[0 if hint is None else 1]]),
                    "attention_mask": torch.ones(1, 1, dtype=torch.long)}

        model = TinyModel()
        policy = TorchPolicy(model, torch.optim.SGD(model.parameters(), lr=.1), encode,
                             lambda tokens: int(tokens[0]), max_grad_norm=10)
        history = History("task", ("screen",), ())
        action = policy.act(history)
        teacher = policy.freeze()
        # Re-scoring must not call generate again, even with a hint.
        teacher.model.generate = lambda **kwargs: self.fail("Teacher generated a repair")
        target = teacher.target(history, action, "repair hint", 0)
        self.assertEqual(encoded_hints, [None, "repair hint", None])
        self.assertEqual(target.action, action)
        self.assertEqual(target.payload.tokens.tolist(), [2])
        self.assertEqual(target.payload.student_inputs["input_ids"].item(), 0)
        cached = target.payload.teacher_log_probs.clone()
        teacher_weights = teacher.model.table.clone()
        before = policy._loss(target).item()
        policy.update(target)
        self.assertLess(policy._loss(target).item(), before)
        torch.testing.assert_close(teacher.model.table, teacher_weights)
        torch.testing.assert_close(target.payload.teacher_log_probs, cached)
        self.assertFalse(target.payload.teacher_log_probs.requires_grad)
        with self.assertRaises(RuntimeError):
            teacher.update(target)


if __name__ == "__main__":
    unittest.main()
