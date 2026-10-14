import unittest

import pyproj  # noqa: F401 - load PROJ before CUDA extension DLLs on Windows
import torch
from torch.nn import functional as F

from pointspace.models import build_model
from pointspace.datasets.utils import point_collate_fn
from pointspace.utils.config import Config


class TestConcertoLitePT(unittest.TestCase):
    def test_two_sample_collate_keeps_patch_indices_local_to_teacher(self):
        sample = dict(
            view1_coord=torch.zeros(3, 3),
            view1_offset=torch.tensor([3], dtype=torch.int32),
            view2_coord=torch.zeros(2, 3),
            view2_offset=torch.tensor([2], dtype=torch.int32),
            dino_feature=torch.randn(2, 1024),
            dino_offset=torch.tensor([2], dtype=torch.long),
            image_patch_index=torch.tensor([0, 1, -1]),
            image_valid=torch.tensor([True, True, False]),
        )
        batch = point_collate_fn([sample, sample])
        self.assertEqual(batch["view1_offset"].tolist(), [3, 6])
        self.assertEqual(batch["view2_offset"].tolist(), [2, 4])
        self.assertEqual(batch["dino_offset"].tolist(), [2, 4])
        self.assertEqual(batch["image_patch_index"].tolist(),
                         [0, 1, -1, 2, 3, -1])

    def test_config_matches_cvrl_backbone(self):
        config = Config.fromfile(
            "configs/baselines/swiss/pretrain-concerto-litept-v1m4-100.py"
        )
        cvrl = Config.fromfile(
            "configs/cvrl/pretrain-cvrl-v1m1-litept-v1m4-swiss-100.py"
        )
        self.assertEqual(dict(config.model.backbone),
                         dict(cvrl.model.backbone.return_backbone))
        self.assertEqual(config.grid_size, cvrl.grid_size)
        self.assertEqual(config.feature_keys, cvrl.feature_keys)
        self.assertEqual(config.data.train.type, "ConcertoLasPairDataset")
        self.assertEqual(config.model.teacher_channels, 1024)

    def test_joint_losses_and_patch_mean_equivalence(self):
        if not torch.cuda.is_available():
            self.skipTest("LitePT requires CUDA")
        config = Config.fromfile(
            "configs/baselines/swiss/pretrain-concerto-litept-v1m4-100.py"
        )
        model = build_model(dict(config.model)).cuda().train()
        n = 1024
        torch.manual_seed(17)
        origin = torch.rand(n, 3, device="cuda") * 12
        intensity = torch.rand(n, 1, device="cuda")
        batch = {}
        for number in (1, 2):
            coord = origin + torch.randn_like(origin) * .01
            batch.update({
                f"view{number}_origin_coord": origin,
                f"view{number}_coord": coord,
                f"view{number}_grid_coord": (coord / .25).floor().int(),
                f"view{number}_feat": torch.cat((coord, intensity), dim=1),
                f"view{number}_offset": torch.tensor([n], device="cuda"),
            })
        for number in range(1, 5):
            batch.update({
                f"local{number}_origin_coord": origin[:256],
                f"local{number}_coord": coord[:256],
                f"local{number}_grid_coord": (coord[:256] / .25).floor().int(),
                f"local{number}_feat": torch.cat((coord[:256], intensity[:256]), dim=1),
                f"local{number}_offset": torch.tensor([256], device="cuda"),
            })
        index = torch.arange(n, device="cuda") % 64
        valid = torch.arange(n, device="cuda") % 4 != 0
        index[~valid] = -1
        batch.update(image_patch_index=index, image_valid=valid,
                     dino_feature=torch.randn(64, 1024, device="cuda"))
        features = torch.randn(n, 72, device="cuda", requires_grad=True)
        cross, patch_count = model._cross_modal_loss(features, batch)
        self.assertEqual(patch_count, 48)
        used, inverse, counts = torch.unique(index[valid], sorted=True,
                                             return_inverse=True,
                                             return_counts=True)
        point_projection = model.patch_projector(features[valid])
        pooled_prediction = torch.zeros(len(used), 1024, device="cuda")
        pooled_prediction.index_add_(0, inverse, point_projection)
        pooled_prediction /= counts[:, None]
        pooled_feature = torch.zeros(len(used), 72, device="cuda")
        pooled_feature.index_add_(0, inverse, features[valid])
        pooled_feature /= counts[:, None]
        self.assertTrue(torch.allclose(
            pooled_prediction, model.patch_projector(pooled_feature), atol=2e-5
        ))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(torch.isfinite(output["intra"]))
        self.assertTrue(torch.isfinite(output["cross"]))
        self.assertEqual(int(output["supervised_patches"]), 48)
        output["loss"].backward()
        self.assertGreater(float(model.student["backbone"].embedding.stem.conv.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.patch_projector.weight.grad.abs().sum()), 0)
        self.assertIsNone(batch["dino_feature"].grad)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feature = model({key[6:]: value for key, value in batch.items()
                             if key.startswith("view1_")},
                            return_point_feature=True,
                            feature_source="student")["point_feature"]
        self.assertEqual(tuple(feature.shape), (n, 72))
        no_image = dict(batch, image_valid=torch.zeros_like(valid))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            image_free = model(no_image)
        self.assertEqual(int(image_free["supervised_patches"]), 0)
        self.assertTrue(torch.allclose(image_free["loss"], image_free["intra"]))


if __name__ == "__main__":
    unittest.main()
