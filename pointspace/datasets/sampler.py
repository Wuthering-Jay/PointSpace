"""
Distributed Weighted Sampler

Combines WeightedRandomSampler with DistributedSampler so that
each rank gets a different, weight-biased subset of the dataset.
"""

import math
import numpy as np
import torch
from torch.utils.data import Sampler

import pointspace.utils.comm as comm


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
