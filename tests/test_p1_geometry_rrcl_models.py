import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from pointspace.models.backbone.geometry_ssl.geometry_ssl_v1m1 import GeometrySSL
from pointspace.models.backbone.rrcl.rrcl_v1m1 import RRCL


class IdentityPointBackbone(nn.Module):
    def forward(self, point):
        return point


def batch():
    torch.manual_seed(7)
    return {
        "coord": torch.randn(12, 3),
        "feat": torch.randn(12, 8, requires_grad=True),
        "offset": torch.tensor([12]),
        "return_role": torch.tensor([0, 1, 2, 3] * 3),
        "geom_valid": torch.ones(12, dtype=torch.bool),
        "normal": torch.nn.functional.normalize(torch.randn(12, 3), dim=-1),
        "local_shape": torch.rand(12, 6),
    }


class TestP1Models(unittest.TestCase):
    def test_geometry_objectives_backward_and_feature_export(self):
        for objective in ("normal", "local_shape"):
            with self.subTest(objective=objective):
                with patch("pointspace.models.backbone.geometry_ssl.geometry_ssl_v1m1.build_model",
                           return_value=IdentityPointBackbone()):
                    model = GeometrySSL(backbone={}, backbone_out_channels=8,
                                        objective=objective)
                data = batch()
                output = model(data)
                self.assertTrue(torch.isfinite(output["loss"]))
                output["loss"].backward()
                self.assertIsNotNone(data["feat"].grad)
                feature = model(data, return_point_feature=True)["point_feature"]
                self.assertEqual(tuple(feature.shape), (12, 8))

    def test_role_does_not_query_knn(self):
        with patch("pointspace.models.backbone.rrcl.rrcl_v1m1.build_model",
                   return_value=IdentityPointBackbone()):
            model = RRCL(backbone={}, backbone_out_channels=8, objective="role",
                         criteria=[dict(type="CrossEntropyLoss", ignore_index=-1)])
        self.assertIsNone(model.context_head)
        with patch("pointspace.models.backbone.rrcl.rrcl_v1m1.build_knn_return_context",
                   side_effect=AssertionError("role-only queried KNN")):
            output = model(batch())
        self.assertIn("loss/role", output)
        self.assertNotIn("loss/ctx", output)
        output["loss"].backward()

    def test_context_has_no_role_head(self):
        with patch("pointspace.models.backbone.rrcl.rrcl_v1m1.build_model",
                   return_value=IdentityPointBackbone()):
            model = RRCL(backbone={}, backbone_out_channels=8, objective="context",
                         criteria=[])
        self.assertIsNone(model.role_head)
        target = torch.full((12, 4), .25)
        valid = torch.ones(12, dtype=torch.bool)
        with patch("pointspace.models.backbone.rrcl.rrcl_v1m1.build_knn_return_context",
                   return_value=(target, valid)):
            output = model(batch())
        self.assertIn("loss/ctx", output)
        self.assertNotIn("loss/role", output)
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()


if __name__ == "__main__":
    unittest.main()
