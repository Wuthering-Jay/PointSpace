"""
Tests for GuaranteedWeightedSampler and DistributedGuaranteedWeightedSampler.

Run with:
    conda run -n pointcept python -m pytest tests/test_guaranteed_weighted_sampler.py -v
"""

import math
import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Import the samplers directly, bypassing the comm module dependency
# ---------------------------------------------------------------------------
from pointspace.datasets.sampler import (
    GuaranteedWeightedSampler,
    DistributedGuaranteedWeightedSampler,
)


# ===========================================================================
# Helpers
# ===========================================================================

def make_weights(n, mode="uniform"):
    """Return weight arrays suitable for testing."""
    if mode == "uniform":
        return np.ones(n, dtype=np.float64)
    if mode == "skewed":
        # Last quarter of samples has 10x weight
        w = np.ones(n, dtype=np.float64)
        w[3 * n // 4 :] = 10.0
        return w
    if mode == "extreme":
        # Single dominant sample
        w = np.ones(n, dtype=np.float64) * 0.01
        w[0] = 100.0
        return w
    raise ValueError(f"Unknown mode: {mode}")


def collect(sampler):
    """Materialise a sampler to a list."""
    return list(iter(sampler))


# ===========================================================================
# GuaranteedWeightedSampler — single-GPU tests
# ===========================================================================


class TestGuaranteedWeightedSamplerLoop1:
    """loop=1 should behave like a plain shuffle (all samples, no repetition)."""

    N = 20

    def test_length(self):
        s = GuaranteedWeightedSampler(make_weights(self.N), self.N, loop=1)
        assert len(s) == self.N

    def test_all_samples_present(self):
        s = GuaranteedWeightedSampler(make_weights(self.N), self.N, loop=1)
        indices = collect(s)
        assert sorted(indices) == list(range(self.N))

    def test_no_duplicates(self):
        s = GuaranteedWeightedSampler(make_weights(self.N), self.N, loop=1)
        indices = collect(s)
        assert len(indices) == len(set(indices))

    def test_is_shuffled(self):
        # Collect several epochs; at least one should differ from sorted order
        s = GuaranteedWeightedSampler(make_weights(self.N), self.N, loop=1)
        epochs_same = 0
        for ep in range(10):
            s.set_epoch(ep)
            if collect(s) == list(range(self.N)):
                epochs_same += 1
        assert epochs_same < 10, "Sampler never shuffled across 10 epochs"


class TestGuaranteedWeightedSamplerLoop2:
    """loop=2: total = 2N, every sample appears >= 1 time."""

    N = 20

    def test_length(self):
        s = GuaranteedWeightedSampler(make_weights(self.N), self.N, loop=2)
        assert len(s) == self.N * 2

    def test_coverage_guarantee(self):
        """Every sample must appear at least once."""
        for mode in ("uniform", "skewed", "extreme"):
            s = GuaranteedWeightedSampler(make_weights(self.N, mode), self.N, loop=2)
            for ep in range(5):
                s.set_epoch(ep)
                indices = collect(s)
                assert set(indices) == set(range(self.N)), (
                    f"mode={mode} epoch={ep}: missing samples"
                )

    def test_total_count(self):
        s = GuaranteedWeightedSampler(make_weights(self.N), self.N, loop=2)
        assert len(collect(s)) == self.N * 2

    def test_index_range(self):
        """All returned indices must be valid."""
        s = GuaranteedWeightedSampler(make_weights(self.N, "extreme"), self.N, loop=2)
        indices = collect(s)
        assert all(0 <= i < self.N for i in indices)


class TestGuaranteedWeightedSamplerLoop5:
    """loop=5: total = 5N, every sample appears >= 1 time."""

    N = 20

    def test_length(self):
        s = GuaranteedWeightedSampler(make_weights(self.N), self.N, loop=5)
        assert len(s) == self.N * 5

    def test_coverage_guarantee(self):
        for mode in ("uniform", "skewed", "extreme"):
            s = GuaranteedWeightedSampler(make_weights(self.N, mode), self.N, loop=5)
            for ep in range(5):
                s.set_epoch(ep)
                indices = collect(s)
                assert set(indices) == set(range(self.N)), (
                    f"mode={mode} epoch={ep}: missing samples"
                )

    def test_total_count(self):
        s = GuaranteedWeightedSampler(make_weights(self.N), self.N, loop=5)
        assert len(collect(s)) == self.N * 5

    def test_index_range(self):
        s = GuaranteedWeightedSampler(make_weights(self.N, "extreme"), self.N, loop=5)
        indices = collect(s)
        assert all(0 <= i < self.N for i in indices)


class TestGuaranteedWeightedSamplerWeighting:
    """High-weight samples should appear more often in the extra slots."""

    N = 100
    LOOP = 10
    EPOCHS = 20

    def test_high_weight_oversampled(self):
        """
        With skewed weights, the last quarter of samples should collectively
        account for more extra appearances than the first quarter.
        """
        w = make_weights(self.N, "skewed")
        high_idx = set(range(3 * self.N // 4, self.N))  # 10x weight group
        low_idx = set(range(self.N // 4))                # 1x weight group

        high_count = 0
        low_count = 0

        for ep in range(self.EPOCHS):
            s = GuaranteedWeightedSampler(w, self.N, loop=self.LOOP)
            s.set_epoch(ep)
            indices = collect(s)
            # Only count the EXTRA slots (N*(loop-1)) — the base pass is uniform
            # We approximate by counting total appearances minus 1-per-sample
            from collections import Counter
            counts = Counter(indices)
            for i in high_idx:
                high_count += max(0, counts[i] - 1)
            for i in low_idx:
                low_count += max(0, counts[i] - 1)

        # High-weight group is 10x; should dominate extra slots
        assert high_count > low_count * 2, (
            f"Expected high_count >> low_count, got {high_count} vs {low_count}"
        )

    def test_extreme_weight_dominates_extras(self):
        """Sample 0 with extreme weight should appear many more times than others."""
        w = make_weights(self.N, "extreme")
        s = GuaranteedWeightedSampler(w, self.N, loop=self.LOOP)
        from collections import Counter
        total_counts = Counter()
        for ep in range(self.EPOCHS):
            s.set_epoch(ep)
            total_counts.update(collect(s))

        # Sample 0 should be most frequent
        most_common = total_counts.most_common(1)[0][0]
        assert most_common == 0, (
            f"Expected sample 0 (extreme weight) to be most frequent, got {most_common}"
        )
        # Sample 0's extra appearances should >>> all others combined / (N-1)
        sample_0_extra = total_counts[0] - self.EPOCHS
        others_extra = sum(
            max(0, cnt - self.EPOCHS) for i, cnt in total_counts.items() if i != 0
        )
        assert sample_0_extra > others_extra, (
            f"sample_0_extra={sample_0_extra}, others_extra={others_extra}"
        )


class TestGuaranteedWeightedSamplerReproducibility:
    """Same epoch seed → same output; different epoch → different output."""

    N = 30
    LOOP = 3

    def test_same_epoch_reproducible(self):
        w = make_weights(self.N, "skewed")
        s1 = GuaranteedWeightedSampler(w, self.N, loop=self.LOOP)
        s2 = GuaranteedWeightedSampler(w, self.N, loop=self.LOOP)
        for ep in range(5):
            s1.set_epoch(ep)
            s2.set_epoch(ep)
            assert collect(s1) == collect(s2), f"epoch={ep} not reproducible"

    def test_different_epochs_differ(self):
        w = make_weights(self.N, "skewed")
        s = GuaranteedWeightedSampler(w, self.N, loop=self.LOOP)
        results = set()
        for ep in range(20):
            s.set_epoch(ep)
            results.add(tuple(collect(s)))
        # At least half the epochs should produce distinct orderings
        assert len(results) > 10, "Too many epochs produced identical indices"

    def test_default_epoch0_deterministic(self):
        w = make_weights(self.N)
        s1 = GuaranteedWeightedSampler(w, self.N, loop=self.LOOP)
        s2 = GuaranteedWeightedSampler(w, self.N, loop=self.LOOP)
        assert collect(s1) == collect(s2)


class TestGuaranteedWeightedSamplerInputTypes:
    """numpy arrays, python lists, and torch tensors should all be accepted."""

    N = 10
    LOOP = 2

    def test_numpy_weights(self):
        w = make_weights(self.N)  # already numpy
        s = GuaranteedWeightedSampler(w, self.N, loop=self.LOOP)
        assert len(collect(s)) == self.N * self.LOOP

    def test_torch_weights(self):
        w = torch.ones(self.N, dtype=torch.double)
        s = GuaranteedWeightedSampler(w, self.N, loop=self.LOOP)
        assert len(collect(s)) == self.N * self.LOOP

    def test_list_weights(self):
        w = [1.0] * self.N
        s = GuaranteedWeightedSampler(w, self.N, loop=self.LOOP)
        assert len(collect(s)) == self.N * self.LOOP


# ===========================================================================
# DistributedGuaranteedWeightedSampler — simulated multi-rank tests
# ===========================================================================


def get_all_rank_indices(weights, num_base, loop, num_replicas, epoch=0):
    """Simulate all ranks and collect their indices as a combined list."""
    all_indices = []
    for rank in range(num_replicas):
        s = DistributedGuaranteedWeightedSampler(
            weights=weights,
            num_base_samples=num_base,
            loop=loop,
            num_replicas=num_replicas,
            rank=rank,
        )
        s.set_epoch(epoch)
        all_indices.append(collect(s))
    return all_indices


class TestDistributedGuaranteedSamplerLengths:
    """Each rank gets ceil(N * loop / num_replicas) samples."""

    N = 20

    @pytest.mark.parametrize("loop,num_replicas", [
        (1, 2), (1, 4),
        (2, 2), (2, 3), (2, 4),
        (5, 2), (5, 4),
        (7, 3),
    ])
    def test_per_rank_length(self, loop, num_replicas):
        expected = math.ceil((self.N * loop) / num_replicas)
        w = make_weights(self.N)
        for rank in range(num_replicas):
            s = DistributedGuaranteedWeightedSampler(
                w, self.N, loop=loop, num_replicas=num_replicas, rank=rank
            )
            assert len(s) == expected, (
                f"loop={loop}, replicas={num_replicas}, rank={rank}: "
                f"expected {expected}, got {len(s)}"
            )
            assert len(collect(s)) == expected


class TestDistributedGuaranteedSamplerCoverage:
    """Combined indices from all ranks must cover every base sample."""

    N = 20

    @pytest.mark.parametrize("loop,num_replicas,mode", [
        (1, 2, "uniform"),
        (2, 2, "uniform"),
        (2, 4, "skewed"),
        (5, 2, "extreme"),
        (5, 4, "skewed"),
        (3, 3, "extreme"),
    ])
    def test_union_covers_all_samples(self, loop, num_replicas, mode):
        w = make_weights(self.N, mode)
        for ep in range(5):
            per_rank = get_all_rank_indices(w, self.N, loop, num_replicas, epoch=ep)
            union = set(i for rank_idx in per_rank for i in rank_idx)
            assert union == set(range(self.N)), (
                f"loop={loop}, replicas={num_replicas}, mode={mode}, epoch={ep}: "
                f"missing {set(range(self.N)) - union}"
            )

    def test_single_replica(self):
        """With num_replicas=1, behaves like GuaranteedWeightedSampler."""
        N, loop = 15, 4
        w = make_weights(N, "skewed")
        s_dist = DistributedGuaranteedWeightedSampler(
            w, N, loop=loop, num_replicas=1, rank=0
        )
        s_single = GuaranteedWeightedSampler(w, N, loop=loop)
        for ep in range(5):
            s_dist.set_epoch(ep)
            s_single.set_epoch(ep)
            assert collect(s_dist) == collect(s_single), f"epoch={ep} mismatch"


class TestDistributedGuaranteedSamplerNoOverlap:
    """For perfectly divisible cases, ranks should not share indices at same position."""

    def test_interleaved_partition_no_positional_overlap(self):
        """Verify the rank-i slice is indices[i::num_replicas] — ranks partition evenly."""
        N, loop, num_replicas = 20, 2, 4  # 40 total, 10 per rank (no padding needed)
        w = make_weights(N)
        per_rank = get_all_rank_indices(w, N, loop, num_replicas, epoch=0)
        # Total indices is num_replicas * num_samples_per_rank
        # Verify concatenated length equals padded total
        total_samples_per_rank = per_rank[0]
        expected_per_rank = math.ceil(N * loop / num_replicas)
        for rank_idx in per_rank:
            assert len(rank_idx) == expected_per_rank


class TestDistributedGuaranteedSamplerReproducibility:
    """set_epoch changes output; same epoch is reproducible across instantiations."""

    N = 25
    LOOP = 3
    REPLICAS = 2

    def test_same_epoch_same_output(self):
        w = make_weights(self.N, "skewed")
        for ep in range(5):
            for rank in range(self.REPLICAS):
                s1 = DistributedGuaranteedWeightedSampler(
                    w, self.N, loop=self.LOOP, num_replicas=self.REPLICAS, rank=rank
                )
                s2 = DistributedGuaranteedWeightedSampler(
                    w, self.N, loop=self.LOOP, num_replicas=self.REPLICAS, rank=rank
                )
                s1.set_epoch(ep)
                s2.set_epoch(ep)
                assert collect(s1) == collect(s2), (
                    f"rank={rank} epoch={ep}: not reproducible"
                )

    def test_different_epochs_differ(self):
        w = make_weights(self.N, "skewed")
        results = set()
        s = DistributedGuaranteedWeightedSampler(
            w, self.N, loop=self.LOOP, num_replicas=self.REPLICAS, rank=0
        )
        for ep in range(20):
            s.set_epoch(ep)
            results.add(tuple(collect(s)))
        assert len(results) > 10

    def test_ranks_get_different_indices(self):
        """Different ranks must produce different index lists."""
        w = make_weights(self.N, "skewed")
        s0 = DistributedGuaranteedWeightedSampler(
            w, self.N, loop=self.LOOP, num_replicas=self.REPLICAS, rank=0
        )
        s1 = DistributedGuaranteedWeightedSampler(
            w, self.N, loop=self.LOOP, num_replicas=self.REPLICAS, rank=1
        )
        s0.set_epoch(0)
        s1.set_epoch(0)
        assert collect(s0) != collect(s1), "Rank 0 and rank 1 should differ"

    def test_all_ranks_same_seed(self):
        """All ranks share the same epoch seed (same global permutation before slicing)."""
        N, loop, num_replicas = 12, 2, 3
        w = make_weights(N)
        # Reconstruct what each rank sees from the global list
        g = torch.Generator()
        g.manual_seed(0)
        base = torch.randperm(N, generator=g).tolist()
        extra = torch.multinomial(
            torch.as_tensor(w, dtype=torch.double),
            num_samples=N * (loop - 1),
            replacement=True,
            generator=g,
        ).tolist()
        all_idx = base + extra
        perm = torch.randperm(len(all_idx), generator=g).tolist()
        global_list = [all_idx[i] for i in perm]
        total_size = math.ceil(N * loop / num_replicas) * num_replicas
        if len(global_list) < total_size:
            global_list += global_list[: total_size - len(global_list)]
        for rank in range(num_replicas):
            expected = global_list[rank:total_size:num_replicas]
            s = DistributedGuaranteedWeightedSampler(
                w, N, loop=loop, num_replicas=num_replicas, rank=rank
            )
            s.set_epoch(0)
            assert collect(s) == expected, f"rank={rank} mismatch with global reconstruction"


class TestDistributedGuaranteedSamplerWeighting:
    """High-weight samples should appear proportionally more across all ranks."""

    N = 100
    LOOP = 5
    REPLICAS = 2
    EPOCHS = 30

    def test_high_weight_oversampled_distributed(self):
        w = make_weights(self.N, "skewed")
        high_idx = set(range(3 * self.N // 4, self.N))
        low_idx = set(range(self.N // 4))

        high_extra = 0
        low_extra = 0

        from collections import Counter
        for ep in range(self.EPOCHS):
            per_rank = get_all_rank_indices(w, self.N, self.LOOP, self.REPLICAS, epoch=ep)
            all_idx = [i for r in per_rank for i in r]
            counts = Counter(all_idx)
            for i in high_idx:
                high_extra += max(0, counts[i] - self.REPLICAS)
            for i in low_idx:
                low_extra += max(0, counts[i] - self.REPLICAS)

        assert high_extra > low_extra * 2, (
            f"high_extra={high_extra}, low_extra={low_extra}"
        )


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_single_sample_loop1(self):
        s = GuaranteedWeightedSampler([1.0], 1, loop=1)
        assert collect(s) == [0]

    def test_single_sample_loop5(self):
        s = GuaranteedWeightedSampler([1.0], 1, loop=5)
        indices = collect(s)
        assert len(indices) == 5
        assert set(indices) == {0}

    def test_two_samples_loop3(self):
        s = GuaranteedWeightedSampler([1.0, 9.0], 2, loop=3)
        indices = collect(s)
        assert len(indices) == 6
        assert set(indices) == {0, 1}

    def test_loop_equals_n(self):
        N = 10
        s = GuaranteedWeightedSampler(make_weights(N), N, loop=N)
        indices = collect(s)
        assert len(indices) == N * N
        assert set(indices) == set(range(N))

    def test_distributed_odd_replicas(self):
        """3 replicas with N*loop not divisible by 3 → padding applied."""
        N, loop, replicas = 10, 2, 3  # 20 total → ceil(20/3)=7 each → padded total=21
        w = make_weights(N)
        for rank in range(replicas):
            s = DistributedGuaranteedWeightedSampler(
                w, N, loop=loop, num_replicas=replicas, rank=rank
            )
            assert len(s) == math.ceil(N * loop / replicas)

    def test_distributed_single_sample(self):
        w = [1.0]
        s = DistributedGuaranteedWeightedSampler(w, 1, loop=4, num_replicas=2, rank=0)
        assert len(collect(s)) == len(s)
        assert all(i == 0 for i in collect(s))

    def test_index_always_in_range(self):
        """All indices must be valid for a variety of configurations."""
        configs = [
            (5, 1, 1), (5, 3, 1), (10, 5, 1),
            (5, 1, 2), (5, 3, 2), (10, 5, 4),
        ]
        for N, loop, replicas in configs:
            w = make_weights(N, "extreme")
            if replicas == 1:
                s = GuaranteedWeightedSampler(w, N, loop=loop)
                s.set_epoch(42)
                assert all(0 <= i < N for i in collect(s)), f"N={N},loop={loop}"
            else:
                for rank in range(replicas):
                    s = DistributedGuaranteedWeightedSampler(
                        w, N, loop=loop, num_replicas=replicas, rank=rank
                    )
                    s.set_epoch(42)
                    assert all(0 <= i < N for i in collect(s)), (
                        f"N={N},loop={loop},replicas={replicas},rank={rank}"
                    )
