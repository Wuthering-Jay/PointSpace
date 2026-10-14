import tempfile
import unittest
from pathlib import Path

import pyproj  # noqa: F401 - establish native DLL load order on Windows
import laspy
import numpy as np
import torch

from pointspace.datasets.las import SonataLasPairDataset
from pointspace.models import build_model
from pointspace.utils.config import Config
from pointspace.engines.hooks.misc import BackboneCheckpointExporter


class TestSonataLitePT(unittest.TestCase):
    def test_chunked_assignment_matches_dense_loss_and_gradient(self):
        from copy import deepcopy
        from pointspace.models.backbone.sonata.sonata_litept_v1m1 import (
            PrototypeHead, SonataLitePT)
        model = SonataLitePT.__new__(SonataLitePT)
        torch.nn.Module.__init__(model)
        head = PrototypeHead(8, 16, 8, 12)
        model.student = torch.nn.ModuleDict({"mask_head": head})
        model.teacher = torch.nn.ModuleDict({"mask_head": deepcopy(head)})
        model.head_chunk_size = 3
        model.teacher_temp = 0.07
        model.student_temp = 0.1
        model.sinkhorn_iterations = 3
        torch.manual_seed(17)
        student = torch.randn(13, 8, requires_grad=True)
        teacher = torch.randn(13, 8)
        index = torch.arange(13)
        chunked = model._assignment_loss(student, teacher, index, index, "mask_head")
        with torch.no_grad():
            target = model._sinkhorn(model.teacher["mask_head"](teacher), .07, 3)
        dense = -(target * torch.log_softmax(
            model.student["mask_head"](student).float() / .1, dim=-1)).sum(-1).mean()
        self.assertTrue(torch.allclose(chunked, dense, atol=1e-6))
        chunk_grad = torch.autograd.grad(chunked, student, retain_graph=True)[0]
        dense_grad = torch.autograd.grad(dense, student)[0]
        self.assertTrue(torch.allclose(chunk_grad, dense_grad, atol=1e-5))

    def test_backbone_export_rewrites_student_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            model = torch.nn.Module()
            model.student = torch.nn.Module()
            model.student.backbone = torch.nn.Linear(4, 3)
            model.student.head = torch.nn.Linear(3, 2)
            path = Path(folder)
            (path / "model").mkdir()
            exporter = BackboneCheckpointExporter(
                filename="test.pth", source_prefix="student.backbone."
            )
            exporter.trainer = type("Trainer", (), {
                "model": model,
                "cfg": type("Cfg", (), {"save_path": str(path)})(),
                "logger": type("Logger", (), {"info": lambda *args: None})(),
            })()
            exporter.after_train()
            state = torch.load(path / "model" / "test.pth", weights_only=False)["state_dict"]
            self.assertEqual(set(state), {"backbone.weight", "backbone.bias"})

    def test_config_and_las_views(self):
        baseline = Config.fromfile(
            "configs/baselines/swiss/pretrain-sonata-litept-v1m4-100.py"
        )
        cvrl = Config.fromfile(
            "configs/cvrl/pretrain-cvrl-v1m1-litept-v1m4-swiss-100.py"
        )
        self.assertEqual(dict(baseline.model.backbone),
                         dict(cvrl.model.backbone.return_backbone))
        self.assertEqual(baseline.feature_keys, cvrl.feature_keys)
        self.assertEqual(baseline.grid_size, cvrl.grid_size)
        # Section 4.5 baselines share a 20-epoch convergence budget.
        self.assertEqual(baseline.epoch, 20)
        self.assertEqual(baseline.data.train.type, "SonataLasPairDataset")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tile.las"
            xyz = np.random.default_rng(4).uniform(0, 10, (300, 3))
            cloud = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
            cloud.x, cloud.y, cloud.z = xyz.T
            cloud.classification = np.full(300, 6, dtype=np.uint8)
            cloud.return_number = np.ones(300, dtype=np.uint8)
            cloud.number_of_returns = np.ones(300, dtype=np.uint8)
            cloud.write(path)
            view = [
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
            dataset = SonataLasPairDataset(data_list=[str(path)],
                                           view1_transform=view, view2_transform=view,
                                           local_transform=view, num_local_views=4)
            sample = dataset[0]
            for prefix in ("view1_", "view2_", "local1_", "local2_",
                           "local3_", "local4_"):
                self.assertTrue({prefix + key for key in
                    ("origin_coord", "coord", "grid_coord", "feat", "offset")}
                    .issubset(sample))

    def test_sinkhorn_and_optimizer_step_ema(self):
        if not torch.cuda.is_available():
            self.skipTest("LitePT/pointops require CUDA")
        cfg = Config.fromfile(
            "configs/baselines/swiss/pretrain-sonata-litept-v1m4-100.py"
        )
        model = build_model(dict(cfg.model)).cuda().train()
        n = 1024
        torch.manual_seed(7)
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
            batch.update({f"local{number}_{key}": value[:256] if key != "offset"
                          else torch.tensor([256], device="cuda")
                          for key, value in {
                              "origin_coord": origin, "coord": coord,
                              "grid_coord": (coord / .25).floor().int(),
                              "feat": torch.cat((coord, intensity), dim=1),
                              "offset": batch["view1_offset"],
                          }.items()})
        self.assertTrue(model._mask({
            "coord": origin, "offset": batch["view1_offset"]
        }).any())
        assignment = model._sinkhorn(torch.randn(32, 64, device="cuda"), .07, 3)
        self.assertTrue(torch.allclose(assignment.sum(-1),
                                       torch.ones(32, device="cuda"), atol=1e-4))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(torch.isfinite(output["roll"]))
        self.assertTrue(torch.isfinite(output["unmask"]))
        self.assertLessEqual(int(output["matched_pairs"]), 2048 * 6)
        self.assertTrue(0 < float(output["matching_fraction"]) <= 1)
        output["loss"].backward()
        self.assertGreater(float(model.student["backbone"].embedding.stem.conv.weight.grad.abs().sum()), 0)
        teacher_before = next(model.teacher.parameters()).clone()
        model.trainer = type("Trainer", (), {"_gradient_accumulation_counter": 1})()
        model.after_step()
        self.assertEqual(model._ema_updates, 0)
        self.assertTrue(torch.equal(teacher_before, next(model.teacher.parameters())))
        model.trainer._gradient_accumulation_counter = 0
        model.trainer._optimizer_step_succeeded = False
        model.after_step()
        self.assertEqual(model._ema_updates, 0)
        model.trainer._optimizer_step_succeeded = True
        model.after_step()
        self.assertEqual(model._ema_updates, 1)
        normal_input = {key[6:]: value for key, value in batch.items()
                        if key.startswith("view1_") and key != "view1_origin_coord"}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feature = model(normal_input, return_point_feature=True,
                            feature_source="teacher")["point_feature"]
        self.assertEqual(tuple(feature.shape), (n, 72))


if __name__ == "__main__":
    unittest.main()
