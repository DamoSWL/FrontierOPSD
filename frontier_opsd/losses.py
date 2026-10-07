"""Reuse the reference sampled K3-IS OPSD objective."""


def distillation_loss(student_log_probs, teacher_log_probs, old_log_probs, mask):
    from verl.trainer.ppo.skillsd_utils import compute_sdl_loss

    if not (student_log_probs.shape == teacher_log_probs.shape == old_log_probs.shape == mask.shape):
        raise ValueError("Teacher, student, old-policy scores and token mask shapes do not align")
    if not mask.isfinite().all() or (mask < 0).any() or mask.sum() <= 0:
        raise ValueError("Token mask must be finite, nonnegative, and contain valid tokens")
    # Exclude padding before calling the unchanged reference function.
    valid = mask > 0
    return compute_sdl_loss(student_log_probs[valid].unsqueeze(0),
        teacher_log_probs[valid].unsqueeze(0), old_log_probs[valid].unsqueeze(0),
        mask[valid].unsqueeze(0))
