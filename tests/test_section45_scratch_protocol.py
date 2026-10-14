"""Guard the first Section 4.5 same-backbone downstream control."""

import unittest
from pathlib import Path

from pointspace.utils.config import Config


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "configs/baselines/dales/semseg-litept-v1m4-scratch-ft.py"
CVRL_FT = ROOT / "configs/cvrl/dales/semseg-cvrl-v1m1-litept-v1m4-dales-ft.py"


class TestSection45ScratchProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = Config.fromfile(str(SCRATCH))
        cls.cvrl = Config.fromfile(str(CVRL_FT))

    def test_same_litept_and_downstream_data(self):
        self.assertEqual(
            dict(self.scratch.model.backbone),
            dict(self.cvrl.model.backbone.return_backbone),
        )
        self.assertEqual(self.scratch.data, self.cvrl.data)
        self.assertEqual(self.scratch.feature_keys, self.cvrl.feature_keys)
        self.assertEqual(self.scratch.grid_size, self.cvrl.grid_size)

    def test_scratch_has_no_pretrained_weight(self):
        self.assertIsNone(self.scratch.weight)
        self.assertFalse(self.scratch.resume)
        self.assertFalse(any(
            hook.type == "CheckpointLoader" for hook in self.scratch.hooks
        ))
        self.assertEqual(self.scratch.model.backbone_out_channels, 72)


if __name__ == "__main__":
    unittest.main()
