import torch

from pointspace.datasets.utils import collate_fn


def test_collate_offsets_grid_sample_inverse_indices():
    first = {
        "coord": torch.zeros((3, 3)),
        "segment": torch.tensor([0, 1, 2]),
        "origin_segment": torch.tensor([0, 1, 2, 1]),
        "inverse": torch.tensor([0, 1, 2, 1]),
    }
    second = {
        "coord": torch.zeros((2, 3)),
        "segment": torch.tensor([3, 4]),
        "origin_segment": torch.tensor([3, 4, 3]),
        "inverse": torch.tensor([0, 1, 0]),
    }

    batch = collate_fn([first, second])

    assert batch["inverse"].tolist() == [0, 1, 2, 1, 3, 4, 3]
    prediction = torch.tensor([10, 11, 12, 13, 14])
    assert prediction[batch["inverse"]].tolist() == [
        10, 11, 12, 11, 13, 14, 13
    ]

