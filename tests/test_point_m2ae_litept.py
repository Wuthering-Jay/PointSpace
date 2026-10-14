import unittest

import pyproj  # noqa: F401 - load PROJ DLLs before CUDA/PointSpace native ops on Windows
import torch

from pointspace.models import build_model
from pointspace.utils.config import Config


class TestPointM2AELitePT(unittest.TestCase):
    def test_swiss_config_matches_cvrl_backbone_and_excludes_labels(self):
        baseline = Config.fromfile(
            "configs/baselines/swiss/pretrain-point-m2ae-litept-v1m4-100.py"
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
        # Section 4.5 baselines use the pre-registered, shorter 20-epoch
        # schedule; architecture and input protocol still match CVRL.
        self.assertEqual(baseline.epoch, 20)
        self.assertFalse(baseline.model.checkpoint_decoder)
        self.assertEqual(baseline.data.train.type, "LasDataset")
        self.assertNotIn("segment", baseline.pretrain_post_transform[-1]["keys"])
        self.assertNotIn("return_role", baseline.pretrain_post_transform[-1]["keys"])

    def test_masked_groups_do_not_enter_encoder_and_backward(self):
        if not torch.cuda.is_available():
            self.skipTest("LitePT/pointops require CUDA")
        cfg = Config.fromfile(
            "configs/cvrl/pretrain-cvrl-v1m1-litept-v1m4-swiss-100.py"
        )
        backbone = dict(cfg.model.backbone.return_backbone)
        model = build_model(dict(
            type="PointM2AE-LitePT-v1m1", backbone=backbone,
            mask_group_size=4.0, fine_group_size=2.0,
            max_coarse_groups=32, max_fine_groups=64,
        )).cuda().train()
        torch.manual_seed(5)
        n = 2048
        coord = torch.rand(n, 3, device="cuda") * torch.tensor(
            [24.0, 24.0, 12.0], device="cuda"
        )
        coord[n // 2:] += torch.tensor([30.0, 0.0, 0.0], device="cuda")
        batch = dict(
            coord=coord,
            grid_coord=(coord / .25).floor().int(),
            feat=torch.cat((coord, torch.rand(n, 1, device="cuda")), dim=1),
            offset=torch.tensor([n // 2, n], device="cuda"),
        )
        filtered, _, target, visible_fraction = model._mask_and_targets(batch)
        self.assertLess(int(filtered["coord"].shape[0]), n)
        self.assertGreater(int(target["coarse"][0].shape[0]), 0)
        self.assertGreater(int(target["fine"][0].shape[0]), 0)
        self.assertTrue(0 < visible_fraction.item() < 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(batch)
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertIsNotNone(model.backbone.embedding.stem.conv.weight.grad)
        self.assertGreater(float(model.backbone.embedding.stem.conv.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.coarse_mask_token.grad.abs().sum()), 0)
        self.assertGreater(float(model.fine_mask_token.grad.abs().sum()), 0)
        self.assertGreater(float(model.coarse_to_fine.weight.grad.abs().sum()), 0)
        model.eval()
        selection = torch.arange(n, device="cuda") % 3 == 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            exported = model(
                dict(batch, feature_point_mask=selection),
                return_point_feature=True,
            )["point_feature"]
        self.assertEqual(tuple(exported.shape), (int(selection.sum()), 72))


if __name__ == "__main__":
    unittest.main()
