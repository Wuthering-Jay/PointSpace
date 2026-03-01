"""
Weighted Samplers with Guaranteed Coverage

Provides samplers that ensure every sample appears at least once per epoch
while oversampling high-weight samples for the remaining slots.

- ``GuaranteedWeightedSampler``: single-GPU version
- ``DistributedGuaranteedWeightedSampler``: multi-GPU version
- ``DistributedWeightedSampler``: legacy multi-GPU weighted sampler (no coverage guarantee)
"""

import math
import numpy as np
import torch
from torch.utils.data import Sampler

import pointspace.utils.comm as comm


class GuaranteedWeightedSampler(Sampler):
    """Weighted sampler that guarantees every sample appears at least once.

    Given *N* base samples and a ``loop`` factor, the total epoch length is
    ``N * loop``.  The first *N* slots are a **shuffled full pass** over all
    samples (coverage guarantee).  The remaining ``(loop - 1) * N`` slots are
    drawn **with replacement** according to *weights*, so higher-weight
    samples are oversampled.

    When ``loop == 1`` this degrades to a plain shuffled pass (no weighting).

    Args:
        weights: Per-sample weight array of length *N* (base dataset size,
            **before** loop).
        num_base_samples: *N*, the number of unique samples (``len(data_list)``).
        loop: Repeat factor.  Total samples per epoch = ``num_base_samples * loop``.
        generator: Optional ``torch.Generator`` for reproducibility.
    """

    def __init__(self, weights, num_base_samples, loop=1, generator=None):
        self.num_base = num_base_samples
        self.loop = loop
        self.total_samples = num_base_samples * loop

        if isinstance(weights, np.ndarray):
            weights = weights.astype(np.float64)
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        assert len(self.weights) == num_base_samples

        self._generator = generator
        self.epoch = 0

    # ------------------------------------------------------------------
    def __iter__(self):
        g = self._generator or torch.Generator()
        g.manual_seed(self.epoch)

        # 1) Full shuffled pass — every sample exactly once
        base_indices = torch.randperm(self.num_base, generator=g).tolist()

        if self.loop <= 1:
            return iter(base_indices)

        # 2) Weighted oversampling for the extra (loop-1)*N slots
        extra = self.total_samples - self.num_base
        extra_indices = torch.multinomial(
            self.weights,
            num_samples=extra,
            replacement=True,
            generator=g,
        ).tolist()

        # 3) Concatenate and shuffle the combined list
        all_indices = base_indices + extra_indices
        perm = torch.randperm(len(all_indices), generator=g).tolist()
        all_indices = [all_indices[i] for i in perm]
        return iter(all_indices)

    def __len__(self):
        return self.total_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedGuaranteedWeightedSampler(Sampler):
    """Distributed version of :class:`GuaranteedWeightedSampler`.

    All ranks share the same seed per epoch so they produce an identical
    global index list, then each rank takes its own interleaved slice
    (same partitioning strategy as ``DistributedSampler``).

    Args:
        weights: Per-sample weight array of length *N*.
        num_base_samples: *N*, #unique samples.
        loop: Repeat factor.
        num_replicas: World size (default: auto).
        rank: Current rank (default: auto).
    """

    def __init__(
        self,
        weights,
        num_base_samples,
        loop=1,
        num_replicas=None,
        rank=None,
    ):
        if num_replicas is None:
            num_replicas = comm.get_world_size()
        if rank is None:
            rank = comm.get_rank()

        self.num_base = num_base_samples
        self.loop = loop
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        if isinstance(weights, np.ndarray):
            weights = weights.astype(np.float64)
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        assert len(self.weights) == num_base_samples

        self.total_size = num_base_samples * loop
        # Pad so total_size is evenly divisible by num_replicas
        self.num_samples = int(math.ceil(self.total_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # 1) Full shuffled pass
        base_indices = torch.randperm(self.num_base, generator=g).tolist()

        if self.loop <= 1:
            all_indices = base_indices
        else:
            extra = self.num_base * self.loop - self.num_base
            extra_indices = torch.multinomial(
                self.weights,
                num_samples=extra,
                replacement=True,
                generator=g,
            ).tolist()
            all_indices = base_indices + extra_indices
            perm = torch.randperm(len(all_indices), generator=g).tolist()
            all_indices = [all_indices[i] for i in perm]

        # Pad to total_size
        if len(all_indices) < self.total_size:
            pad = self.total_size - len(all_indices)
            all_indices += all_indices[:pad]

        # Subsample for this rank
        indices = all_indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedWeightedSampler(Sampler):
    """Distributed sampler that respects per-sample weights.

    Each epoch:
        1.  Draw ``total_size`` indices from ``[0, len(weights))`` with
            probability proportional to *weights* (with replacement).
        2.  Partition the drawn indices across ranks so each rank gets
            ``num_samples`` indices.

    Supports ``set_epoch()`` for reproducible shuffling, same as
    ``DistributedSampler``.

    Args:
        weights: Per-sample weight array (length must match dataset ``__len__``).
        dataset: The dataset (only used for ``len(dataset)``).
        num_replicas: Number of processes (default: world size).
        rank: Current process rank (default: ``comm.get_rank()``).
        replacement: Sample with replacement (default: True).
    """

    def __init__(
        self,
        weights,
        dataset,
        num_replicas=None,
        rank=None,
        replacement=True,
    ):
        if num_replicas is None:
            num_replicas = comm.get_world_size()
        if rank is None:
            rank = comm.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.epoch = 0

        # Convert to float64 tensor for multinomial sampling
        if isinstance(weights, np.ndarray):
            weights = weights.astype(np.float64)
        self.weights = torch.as_tensor(weights, dtype=torch.double)

        self.total_size = len(self.dataset)
        self.num_samples = int(math.ceil(self.total_size / self.num_replicas))
        # Pad total_size so it is evenly divisible
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        # Draw weighted indices for the entire dataset (all ranks share the
        # same seed → same draw → deterministic partitioning).
        indices = torch.multinomial(
            self.weights,
            num_samples=self.total_size,
            replacement=self.replacement,
            generator=g,
        ).tolist()
        # Subsample for this rank
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
