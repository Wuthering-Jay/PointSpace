import tempfile
import unittest
from pathlib import Path

import pyproj  # noqa: F401 - load PROJ DLLs before CUDA/PointSpace native ops on Windows
import numpy as np
import torch
import torch.nn.functional as F

from pointspace.datasets.las import MSCLasPairDataset
from pointspace.datasets.utils import point_collate_fn
from pointspace.models import build_model
from pointspace.utils.config import Config
import laspy


class TestMSCLitePT(unittest.TestCase):
    def test_chunked_infonce_matches_original_dense_pool(self):
        from pointspace.models.backbone.msc.msc_litept_v1m1 import MSCLitePT
        objective = MSCLitePT.__new__(MSCLitePT)
        torch.nn.Module.__init__(objective)
        objective.nce_chunk_size = 3
        objective.temperature = 0.4
        torch.manual_seed(31)
        first = torch.randn(11, 7, requires_grad=True)
        second = torch.randn(13, 7, requires_grad=True)
        anchor = torch.tensor([0, 2, 4, 5, 8, 10])
        positive = torch.tensor([1, 3, 7, 9, 10, 12])
        chunked, _, _, count = objective._contrastive_loss(
            first, second, anchor, positive)
        dense = F.cross_entropy(
            F.normalize(first[anchor], dim=-1) @
            F.normalize(second[positive], dim=-1).T / 0.4,
            torch.arange(count))
        self.assertTrue(torch.allclose(chunked, dense, atol=1e-6))

    def test_swiss_config_matches_cvrl_backbone(self):
        baseline = Config.fromfile(
            "configs/baselines/swiss/pretrain-msc-litept-v1m4-100.py"
        )
        cvrl = Config.fromfile(
            "configs/cvrl/pretrain-cvrl-v1m1-litept-v1m4-swiss-100.py"
        )
        self.assertEqual(
            dict(baseline.model.backbone),
            dict(cvrl.model.backbone.return_backbone),
        )
        self.assertEqual(baseline.feature_keys, cvrl.feature_keys)
        self.assertEqual(baseline.grid_size, cvrl.grid_size)
        # Section 4.5 baselines share a 20-epoch convergence budget.
        self.assertEqual(baseline.epoch, 20)
        self.assertEqual(baseline.data.train.type, "MSCLasPairDataset")
        self.assertEqual(baseline.model.matching_max_pair, 8192)
        self.assertFalse(baseline.model.reconstruct_normal)
        self.assertEqual(baseline.model.view1_mix_prob, 0.8)

    def test_las_pair_excludes_labels_and_return_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tile.las"
            rng = np.random.default_rng(4)
            xyz = rng.uniform(0, 15, (800, 3))
            cloud = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
            cloud.x, cloud.y, cloud.z = xyz.T
            cloud.classification = np.full(800, 6, dtype=np.uint8)
            cloud.return_number = np.ones(800, dtype=np.uint8)
            cloud.number_of_returns = np.ones(800, dtype=np.uint8)
            cloud.write(path)
            transforms = [
                dict(type="EnsureFeature", key="intensity", channels=1,
                     value=0.0, overwrite=True),
                dict(type="Copy", keys_dict={"coord": "origin_coord"}),
                dict(type="Update", keys_dict={
                    "index_valid_keys": ["coord", "intensity", "origin_coord"]
                }),
                dict(type="GridSample", grid_size=.25, hash_type="fnv",
                     mode="train", return_grid_coord=True),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("origin_coord", "coord", "grid_coord"),
                     feat_keys=("coord", "intensity")),
            ]
            dataset = MSCLasPairDataset(
                data_list=[str(path)], view1_transform=transforms,
                view2_transform=transforms,
            )
            sample = dataset[0]
            self.assertEqual(set(sample), {
                "view1_origin_coord", "view1_coord", "view1_grid_coord",
                "view1_feat", "view1_offset", "view2_origin_coord",
                "view2_coord", "view2_grid_coord", "view2_feat", "view2_offset",
            })
            self.assertEqual(sample["view1_feat"].shape[1], 4)
            self.assertEqual(len(point_collate_fn([sample])["view1_offset"]), 1)

    def test_bounded_contrastive_forward_backward_and_export(self):
        if not torch.cuda.is_available():
            self.skipTest("LitePT/pointops require CUDA")
        cfg = Config.fromfile(
            "configs/cvrl/pretrain-cvrl-v1m1-litept-v1m4-swiss-100.py"
        )
        model = build_model(dict(
            type="MSC-LitePT-v1m1",
            backbone=dict(cfg.model.backbone.return_backbone),
            mask_group_size=4.0, mask_rate=.4,
            matching_max_radius=.25, matching_max_pair=128,
            normal_k=16, nce_chunk_size=32,
            reconstruct_normal=True,
        )).cuda().train()
        torch.manual_seed(9)
        n = 2048
        origin = torch.rand(n, 3, device="cuda") * 20
        origin[n // 2:] += torch.tensor([30.0, 0.0, 0.0], device="cuda")
        intensity = torch.rand(n, 1, device="cuda")
        batch = {}
        for number in (1, 2):
            coord = origin + torch.randn_like(origin) * .01
            batch.update({
                f"view{number}_origin_coord": origin,
                f"view{number}_coord": coord,
                f"view{number}_grid_coord": (coord / .25).floor().int(),
                f"view{number}_feat": torch.cat((coord, intensity), dim=1),
                f"view{number}_offset": torch.tensor([n // 2, n], device="cuda"),
            })
        first = {key[6:]: value for key, value in batch.items()
                 if key.startswith("view1_")}
        second = {key[6:]: value for key, value in batch.items()
                  if key.startswith("view2_")}
        mask1, mask2 = model._cross_view_masks(first, second)
        self.assertTrue(mask1.any())
        self.assertTrue(mask2.any())
        self.assertFalse(torch.any(mask1 & mask2))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(torch.isfinite(output["normal"]))
        self.assertTrue(torch.isfinite(output["intensity"]))
        self.assertLessEqual(int(output["matched_pairs"]), 256)
        output["loss"].backward()
        self.assertGreater(float(model.backbone.embedding.stem.conv.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.normal_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.intensity_head.weight.grad.abs().sum()), 0)
        model.eval()
        normal_input = {key[6:]: value for key, value in batch.items()
                        if key.startswith("view1_") and key != "view1_origin_coord"}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feature = model(normal_input, return_point_feature=True)["point_feature"]
        self.assertEqual(tuple(feature.shape), (n, 72))


if __name__ == "__main__":
    unittest.main()
