"""
EZSP Utility Functions

Bridge utilities between PointSpace offset format and NAG/SPT CSR ptr format.

PointSpace format:
    offset: [n1, n1+n2, n1+n2+n3, ...] - cumulative sum of sizes (NO leading zero)

NAG/SPT CSR format:
    ptr: [0, n1, n1+n2, n1+n2+n3, ...] - cumulative sum WITH leading zero
"""

import torch
from torch import Tensor
from typing import Tuple, Optional


def offset_to_ptr(offset: Tensor) -> Tensor:
    """Convert PointSpace offset to CSR ptr format.

    PointSpace: offset = [5, 8, 12] for 3 batches with sizes [5, 3, 4]
    CSR ptr:    ptr    = [0, 5, 8, 12]

    Args:
        offset: 1D tensor of cumulative sizes without leading zero

    Returns:
        1D tensor with leading zero prepended
    """
    zero = torch.zeros(1, device=offset.device, dtype=offset.dtype)
    return torch.cat([zero, offset])


def ptr_to_offset(ptr: Tensor) -> Tensor:
    """Convert CSR ptr format to PointSpace offset.

    CSR ptr:    ptr    = [0, 5, 8, 12]
    PointSpace: offset = [5, 8, 12]

    Args:
        ptr: 1D tensor with leading zero

    Returns:
        1D tensor without leading zero
    """
    return ptr[1:].contiguous()


def sizes_to_ptr(sizes: Tensor) -> Tensor:
    """Convert segment sizes to CSR ptr format.

    sizes = [5, 3, 4] -> ptr = [0, 5, 8, 12]

    Args:
        sizes: 1D tensor of segment sizes

    Returns:
        1D tensor in CSR ptr format
    """
    zero = torch.zeros(1, device=sizes.device, dtype=torch.long)
    return torch.cat([zero, sizes.long()]).cumsum(dim=0)


def ptr_to_sizes(ptr: Tensor) -> Tensor:
    """Convert CSR ptr format to segment sizes.

    ptr = [0, 5, 8, 12] -> sizes = [5, 3, 4]

    Args:
        ptr: 1D tensor in CSR ptr format

    Returns:
        1D tensor of segment sizes
    """
    return ptr[1:] - ptr[:-1]


def sizes_to_offset(sizes: Tensor) -> Tensor:
    """Convert segment sizes to PointSpace offset.

    sizes = [5, 3, 4] -> offset = [5, 8, 12]

    Args:
        sizes: 1D tensor of segment sizes

    Returns:
        1D tensor in PointSpace offset format
    """
    return sizes.long().cumsum(dim=0)


def offset_to_sizes(offset: Tensor) -> Tensor:
    """Convert PointSpace offset to segment sizes.

    offset = [5, 8, 12] -> sizes = [5, 3, 4]

    Args:
        offset: 1D tensor in PointSpace offset format

    Returns:
        1D tensor of segment sizes
    """
    zero = torch.zeros(1, device=offset.device, dtype=offset.dtype)
    return torch.diff(offset, prepend=zero)


def indices_to_ptr(indices: Tensor, num_segments: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """Convert sorted segment indices to CSR ptr format.

    Converts pre-sorted dense indices to CSR format. If indices are not sorted,
    they will be sorted and an order tensor will be returned.

    Args:
        indices: 1D tensor of segment indices (should be dense: 0, 1, 2, ...)
        num_segments: Optional number of segments. If None, inferred from max index.

    Returns:
        Tuple of (ptr, order):
        - ptr: CSR pointer tensor
        - order: Permutation tensor to sort original indices
    """
    device = indices.device
    assert indices.dim() == 1, "Only 1D indices are accepted"
    assert indices.numel() >= 1, "At least one index is required"

    # Sort indices if needed
    if not (indices[:-1] <= indices[1:]).all():
        indices, order = indices.sort()
    else:
        order = torch.arange(indices.shape[0], device=device)

    # Infer num_segments if not provided
    if num_segments is None:
        num_segments = indices.max().item() + 1

    # Convert sorted indices to pointers
    ptr = torch.zeros(num_segments + 1, device=device, dtype=torch.long)
    ones = torch.ones_like(indices)
    ptr.scatter_add_(0, indices + 1, ones)
    ptr = ptr.cumsum(dim=0)

    return ptr, order


def ptr_to_batch(ptr: Tensor) -> Tensor:
    """Convert CSR ptr to batch indices.

    ptr = [0, 5, 8, 12] -> batch = [0,0,0,0,0, 1,1,1, 2,2,2,2]

    Args:
        ptr: CSR pointer tensor

    Returns:
        1D tensor of batch/segment indices for each element
    """
    sizes = ptr_to_sizes(ptr)
    return torch.arange(len(sizes), device=ptr.device, dtype=torch.long).repeat_interleave(sizes)


def batch_to_ptr(batch: Tensor, num_segments: Optional[int] = None) -> Tensor:
    """Convert batch indices to CSR ptr format.

    batch = [0,0,0,0,0, 1,1,1, 2,2,2,2] -> ptr = [0, 5, 8, 12]

    Args:
        batch: 1D tensor of segment indices
        num_segments: Optional number of segments

    Returns:
        CSR pointer tensor
    """
    if num_segments is None:
        num_segments = batch.max().item() + 1
    sizes = torch.bincount(batch, minlength=num_segments)
    return sizes_to_ptr(sizes)


# ============================================================================
# Super index utilities (for hierarchical partitions)
# ============================================================================

def super_index_to_sub_ptr(super_index: Tensor, num_super: Optional[int] = None) -> Tensor:
    """Convert super_index to sub CSR pointers.

    super_index[i] = j means point i belongs to superpoint j.
    Returns ptr where ptr[j]:ptr[j+1] gives indices of points in superpoint j.

    Args:
        super_index: 1D tensor mapping points to superpoints
        num_super: Number of superpoints (optional)

    Returns:
        CSR pointer tensor for accessing sub-elements
    """
    return batch_to_ptr(super_index, num_super)


def compute_super_index(sub_ptr: Tensor) -> Tensor:
    """Compute super_index from sub CSR pointers.

    Inverse of super_index_to_sub_ptr.

    Args:
        sub_ptr: CSR pointer tensor

    Returns:
        super_index tensor
    """
    return ptr_to_batch(sub_ptr)
