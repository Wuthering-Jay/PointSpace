import unittest

import pyproj  # noqa: F401 - load PROJ before native model extensions
import torch

from pointspace.models import build_model
from pointspace.utils.config import Config


class TestSection45DownstreamProtocol(unittest.TestCase):
    def _config(self, mode):
        return Config.fromfile(
            f"configs/baselines/dales/semseg-litept-v1m4-pretrained-{mode}.py"
        )

    def test_all_modes_use_matching_pretrained_litept(self):
        source = Config.fromfile(
            "configs/baselines/swiss/pretrain-sonata-litept-v1m4-100.py"
        )
        for mode in ("lin", "dec", "ft"):
            config = self._config(mode)
            self.assertEqual(config.pretrain_method, "sonata")
            self.assertEqual(config.weight, source.save_path + "/model/backbone_last.pth")
            self.assertEqual(config.feature_keys, source.feature_keys)
            self.assertEqual(config.grid_size, source.grid_size)
            self.assertEqual(config.data.train.required_class,
                             [1, 2, 3, 4, 5, 6, 7, 8])
            self.assertEqual(config.data.train.transform[1].type, "EnsureFeature")
            self.assertTrue(config.data.train.transform[1].overwrite)
            if mode != "dec":
                self.assertEqual(dict(config.model.backbone),
                                 dict(source.model.backbone))
            self.assertTrue(config.hooks[0].required)

    def test_trainable_parameter_boundaries(self):
        linear = build_model(dict(self._config("lin").model))
        linear.train()
        self.assertFalse(linear.backbone.training)
        trainable = {name for name, parameter in linear.named_parameters()
                     if parameter.requires_grad}
        self.assertEqual(trainable, {"seg_head.weight", "seg_head.bias"})
        self.assertIsInstance(linear.seg_head, torch.nn.Linear)

        decoder = build_model(dict(self._config("dec").model))
        decoder.train()
        self.assertFalse(decoder.backbone.embedding.training)
        self.assertFalse(decoder.backbone.enc.training)
        self.assertTrue(decoder.backbone.dec.training)
        self.assertFalse(any(p.requires_grad for p in decoder.backbone.enc.parameters()))
        self.assertTrue(any(p.requires_grad for p in decoder.backbone.dec.parameters()))
        self.assertEqual(decoder.seg_head.in_features, 72)

        finetune = build_model(dict(self._config("ft").model))
        self.assertTrue(any(p.requires_grad for p in finetune.backbone.enc.parameters()))
        self.assertTrue(any(p.requires_grad for p in finetune.backbone.dec.parameters()))
        self.assertEqual(finetune.seg_head.in_features, 72)


if __name__ == "__main__":
    unittest.main()
