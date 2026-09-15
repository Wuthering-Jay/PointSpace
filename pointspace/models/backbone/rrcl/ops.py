"""Target construction utilities for Return Role and Context Learning."""

import torch
import torch.nn.functional as F


SINGLE = 0
FIRST = 1
MIDDLE = 2
LAST = 3
NUM_RETURN_ROLES = 4


def build_return_role(return_number, num_returns, ignore_index=-1):
    """Map LAS return number/count tensors to S/F/M/L class indices."""
    role = torch.full_like(return_number, ignore_index, dtype=torch.long)
    single = num_returns == 1
    first = (num_returns > 1) & (return_number == 1)
    middle = (
        (num_returns > 1)
        & (return_number > 1)
        & (return_number < num_returns)
    )
    last = (num_returns > 1) & (return_number == num_returns)
    # Single must remain a separate first branch because r == 1 == n.
    role[single] = SINGLE
    role[first] = FIRST
    role[middle] = MIDDLE
    role[last] = LAST
    return role


@torch.no_grad()
def build_knn_return_context(
    coord,
    offset,
    return_role,
    k=32,
    ignore_index=-1,
    pointops_module=None,
):
    """Build current-point-set KNN role histograms without the center point."""
    if pointops_module is None:
        import pointops as pointops_module

    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(f"coord must have shape [N, 3], got {tuple(coord.shape)}")
    return_role = return_role.reshape(-1).long()
    if return_role.numel() != coord.shape[0]:
        raise ValueError("return_role and coord must contain the same point count")
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")

    target = coord.new_zeros((coord.shape[0], NUM_RETURN_ROLES), dtype=torch.float32)
    valid_center = torch.zeros(coord.shape[0], dtype=torch.bool, device=coord.device)
    valid_role = (return_role >= 0) & (return_role < NUM_RETURN_ROLES)
    if not valid_role.any():
        # A visual-only cloud (for example synthetic water-surface points)
        # has no RRCL supervision. Avoid constructing a large, unused KNN
        # graph while returning correctly masked zero targets.
        return target, valid_center
    starts = F.pad(offset.long(), (1, 0), value=0)[:-1]

    def fill_histogram(row_slice, neighbor_role, center_role):
        valid_neighbor = (neighbor_role >= 0) & (
            neighbor_role < NUM_RETURN_ROLES
        )
        one_hot = F.one_hot(
            neighbor_role.clamp(0, NUM_RETURN_ROLES - 1),
            num_classes=NUM_RETURN_ROLES,
        ).float()
        one_hot = one_hot * valid_neighbor.unsqueeze(-1)
        denominator = valid_neighbor.sum(dim=1, keepdim=True)
        target[row_slice] = one_hot.sum(dim=1) / denominator.clamp_min(1)
        valid_center[row_slice] = (
            (center_role != ignore_index)
            & (center_role >= 0)
            & (center_role < NUM_RETURN_ROLES)
            & denominator.squeeze(1).gt(0)
        )

    counts = offset.long() - starts

    def query_indices(query_k, query_coord, query_offset):
        idx_only = getattr(pointops_module, "knn_query_idx", None)
        if idx_only is not None:
            return idx_only(query_k, query_coord, query_offset)
        # Compatibility for third-party pointops builds and lightweight tests.
        return pointops_module.knn_query(
            query_k, query_coord, query_offset
        )[0]

    def exclude_center(neighbor_idx, center_idx, neighbor_count):
        """Drop the actual center index, including in zero-distance ties."""
        is_neighbor = neighbor_idx.ne(center_idx[:, None])
        neighbor_rank = is_neighbor.long().cumsum(dim=1) - 1
        keep = is_neighbor & neighbor_rank.lt(neighbor_count)
        return neighbor_idx[keep].reshape(neighbor_idx.shape[0], neighbor_count)

    if counts.numel() > 0 and counts.min().item() > k:
        # Normal training path: one batched CUDA query.  The per-cloud fallback
        # below exists only for batches containing a tile with <= k points.
        neighbor_idx = query_indices(
            k + 1, coord.float().contiguous(), offset.int()
        )
        center_idx = torch.arange(coord.shape[0], device=coord.device)
        neighbor_idx = exclude_center(neighbor_idx.long(), center_idx, k)
        fill_histogram(slice(None), return_role[neighbor_idx], return_role)
        return target, valid_center

    # Fallback: query each cloud independently so tiny tiles remain valid.
    for start_tensor, end_tensor in zip(starts, offset.long()):
        start = int(start_tensor.item())
        end = int(end_tensor.item())
        count = end - start
        k_eff = min(k, count - 1)
        if k_eff <= 0:
            continue
        cloud_coord = coord[start:end].float().contiguous()
        cloud_offset = torch.tensor([count], dtype=torch.int32, device=coord.device)
        neighbor_idx = query_indices(
            k_eff + 1, cloud_coord, cloud_offset
        )
        # pointops orders Euclidean neighbors by distance, hence column zero
        # is the zero-distance query point itself.
        center_idx = torch.arange(count, device=coord.device)
        neighbor_idx = exclude_center(neighbor_idx.long(), center_idx, k_eff)
        neighbor_role = return_role[start:end][neighbor_idx]
        fill_histogram(slice(start, end), neighbor_role, return_role[start:end])
    return target, valid_center
