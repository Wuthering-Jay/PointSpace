"""Gradient-projection primitives used by the CVRL PCGrad trainer."""

import torch


def accumulate_gradient_tuple(accumulated, gradients):
    """Add one task-gradient tuple to an accumulation-window buffer."""
    if accumulated is None:
        return tuple(
            None if gradient is None else gradient.detach().clone()
            for gradient in gradients
        )
    if len(accumulated) != len(gradients):
        raise ValueError("Accumulated and current gradient lengths differ")
    updated = []
    for current, gradient in zip(accumulated, gradients):
        if current is None:
            current = None if gradient is None else gradient.detach().clone()
        elif gradient is not None:
            current.add_(gradient.detach())
        updated.append(current)
    return tuple(updated)


def subtract_gradient_tuples(combined_grad, subtrahend_grad):
    """Recover one accumulated task from combined minus the other task."""
    if len(combined_grad) != len(subtrahend_grad):
        raise ValueError("Combined and task gradient lengths differ")
    recovered = []
    for combined, subtrahend in zip(combined_grad, subtrahend_grad):
        if combined is None:
            recovered.append(None)
        elif subtrahend is None:
            recovered.append(combined)
        else:
            recovered.append(combined - subtrahend)
    return tuple(recovered)


def project_conflicting_gradient_group(visual_grad, return_grad, eps=1e-12):
    """Symmetrically project one stage of two task gradients.

    ``None`` entries represent parameters unused by one task and pass through
    unchanged. Reductions are evaluated in FP32, while returned gradients keep
    their original parameter-gradient dtype.
    """
    reference = next(
        (
            grad
            for pair in zip(visual_grad, return_grad)
            for grad in pair
            if grad is not None
        ),
        None,
    )
    if reference is None:
        raise ValueError("PCGrad group contains no task gradients")

    dot = reference.new_zeros((), dtype=torch.float32)
    visual_norm2 = dot.clone()
    return_norm2 = dot.clone()
    for grad_v, grad_r in zip(visual_grad, return_grad):
        if grad_v is not None:
            visual_norm2 += grad_v.detach().float().square().sum()
        if grad_r is not None:
            return_norm2 += grad_r.detach().float().square().sum()
        if grad_v is not None and grad_r is not None:
            dot += (grad_v.detach().float() * grad_r.detach().float()).sum()

    valid = (visual_norm2 > eps) & (return_norm2 > eps)
    conflict = valid & (dot < 0)
    zero = dot.new_zeros(())
    visual_coefficient = torch.where(
        conflict, dot / return_norm2.clamp_min(eps), zero
    )
    return_coefficient = torch.where(
        conflict, dot / visual_norm2.clamp_min(eps), zero
    )

    projected_visual = []
    projected_return = []
    delta_norm2 = dot.new_zeros(())
    original_sum_norm2 = dot.new_zeros(())
    for grad_v, grad_r in zip(visual_grad, return_grad):
        projected_v = grad_v
        projected_r = grad_r
        if grad_v is not None and grad_r is not None:
            projected_v = grad_v - visual_coefficient.to(grad_v.dtype) * grad_r
            projected_r = grad_r - return_coefficient.to(grad_r.dtype) * grad_v
        projected_visual.append(projected_v)
        projected_return.append(projected_r)

        if grad_v is None and grad_r is None:
            continue
        original = grad_r if grad_v is None else grad_v
        projected = projected_r if projected_v is None else projected_v
        if grad_v is not None and grad_r is not None:
            original = grad_v + grad_r
            projected = projected_v + projected_r
        delta_norm2 += (projected.float() - original.float()).square().sum()
        original_sum_norm2 += original.float().square().sum()

    post_dot = dot.new_zeros(())
    post_visual_norm2 = dot.new_zeros(())
    post_return_norm2 = dot.new_zeros(())
    for grad_v, grad_r in zip(projected_visual, projected_return):
        if grad_v is not None:
            post_visual_norm2 += grad_v.detach().float().square().sum()
        if grad_r is not None:
            post_return_norm2 += grad_r.detach().float().square().sum()
        if grad_v is not None and grad_r is not None:
            post_dot += (grad_v.detach().float() * grad_r.detach().float()).sum()

    visual_norm = visual_norm2.sqrt()
    return_norm = return_norm2.sqrt()
    stats = {
        "pre_cosine": dot / (visual_norm * return_norm).clamp_min(eps),
        "post_cosine": post_dot
        / (post_visual_norm2.sqrt() * post_return_norm2.sqrt()).clamp_min(eps),
        "conflict_rate": conflict.float(),
        "projection_ratio": delta_norm2.sqrt()
        / original_sum_norm2.sqrt().clamp_min(eps),
        "visual_norm": visual_norm,
        "return_norm": return_norm,
        "visual_return_norm_ratio": visual_norm
        / return_norm.clamp_min(eps),
    }
    return projected_visual, projected_return, stats

