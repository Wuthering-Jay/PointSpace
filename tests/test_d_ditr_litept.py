import unittest

import pyproj  # noqa: F401 - load PROJ before native CUDA extension DLLs
import torch

from pointspace.models import build_model
from pointspace.utils.config import Config


class TestDDITRLitePT(unittest.TestCase):
    def test_config_matches_cvrl_backbone(self):
        baseline = Config.fromfile(
            "configs/baselines/swiss/pretrain-d-ditr-litept-v1m4-100.py"
        )
        cvrl = Config.fromfile(
            "configs/cvrl/pretrain-cvrl-v1m1-litept-v1m4-swiss-100.py"
        )
        self.assertEqual(dict(baseline.model.backbone),
                         dict(cvrl.model.backbone.return_backbone))
        self.assertEqual(baseline.grid_size, cvrl.grid_size)
        self.assertEqual(baseline.feature_keys, cvrl.feature_keys)
        self.assertEqual(baseline.data.train.type, "LasImageDataset")
        self.assertEqual(baseline.model.teacher_channels, 1024)
        self.assertEqual(baseline.model.loss_chunk_size, 16384)
        self.assertFalse(baseline.model.checkpoint_loss)
        self.assertTrue(baseline.model.normalize_teacher_once)

    def test_pointwise_cosine_target_and_export(self):
        if not torch.cuda.is_available():
            self.skipTest("LitePT requires CUDA")
        cfg = Config.fromfile(
            "configs/baselines/swiss/pretrain-d-ditr-litept-v1m4-100.py"
        )
        model = build_model(dict(cfg.model)).cuda().train()
        n = 1024
        torch.manual_seed(13)
        coord = torch.rand(n, 3, device="cuda") * 10
        patch_index = torch.arange(n, device="cuda") % 48
        valid = torch.arange(n, device="cuda") % 5 != 0
        patch_index[~valid] = -1
        batch = dict(
            coord=coord,
            grid_coord=(coord / .25).floor().int(),
            feat=torch.cat((coord, torch.rand(n, 1, device="cuda")), dim=1),
            offset=torch.tensor([n], device="cuda"),
            dino_feature=torch.randn(48, 1024, device="cuda"),
            image_patch_index=patch_index,
            image_valid=valid,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertEqual(int(output["supervised_points"]), int(valid.sum()))
        output["loss"].backward()
        self.assertGreater(float(model.backbone.embedding.stem.conv.weight.grad.abs().sum()), 0)
        self.assertIsNone(batch["dino_feature"].grad)
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            features = model(batch, return_point_feature=True)["point_feature"]
        self.assertEqual(tuple(features.shape), (n, 72))
        bad = dict(batch, image_valid=torch.zeros_like(valid))
        with self.assertRaisesRegex(RuntimeError, "no valid"):
            model(bad)


if __name__ == "__main__":
    unittest.main()
