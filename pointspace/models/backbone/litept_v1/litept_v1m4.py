"""LitePT encoder variant exposing a stable hierarchy for HPSD."""

from pointspace.models.builder import MODELS

from ..hpsd.hierarchy import build_encoder_hierarchy
from .litept_v1m3_utonia import LitePT as LitePTM3


@MODELS.register_module("LitePT-v1m4")
class LitePTV1M4(LitePTM3):
    """Reviewed encoder-only LitePT with optional hierarchy output."""

    def __init__(self, *args, enc_mode=True, **kwargs):
        if not enc_mode:
            raise ValueError("LitePT-v1m4 requires enc_mode=True")
        super().__init__(*args, enc_mode=True, **kwargs)

    def forward(self, data_dict, return_hierarchy=False):
        point = super().forward(data_dict)
        if not return_hierarchy:
            return point
        hierarchy = build_encoder_hierarchy(point, expected_levels=self.num_stages)
        return point, hierarchy

