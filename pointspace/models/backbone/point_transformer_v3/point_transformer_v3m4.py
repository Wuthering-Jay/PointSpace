"""PTV3 encoder variant exposing a stable hierarchy for HPSD."""

from pointspace.models.builder import MODELS

from ..hpsd.hierarchy import build_encoder_hierarchy
from .point_transformer_v3m3_utonia import PointTransformerV3 as PointTransformerV3M3


@MODELS.register_module("PT-v3m4")
class PointTransformerV3M4(PointTransformerV3M3):
    """Reviewed encoder-only PTV3 with optional fine-to-coarse hierarchy."""

    def __init__(self, *args, enc_mode=True, **kwargs):
        if not enc_mode:
            raise ValueError("PT-v3m4 requires enc_mode=True")
        super().__init__(*args, enc_mode=True, **kwargs)

    def forward(self, data_dict, return_hierarchy=False):
        point = super().forward(data_dict)
        if not return_hierarchy:
            return point
        hierarchy = build_encoder_hierarchy(point, expected_levels=self.num_stages)
        return point, hierarchy

