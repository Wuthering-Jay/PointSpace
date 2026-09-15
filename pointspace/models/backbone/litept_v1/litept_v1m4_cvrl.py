"""LitePT variant exposing a reusable hierarchy for VPSD and CVRL."""

from pointspace.models.builder import MODELS
from pointspace.models.utils.structure import Point

from ..vpsd.hierarchy import build_encoder_hierarchy
from .litept_v1m3_utonia import LitePT as LitePTM3


def _clone_hierarchy_for_decoder(hierarchy, detach_stages=()):
    """Clone hierarchy containers and selectively detach encoder features."""
    detach_stages = {int(stage) for stage in detach_stages}
    invalid = detach_stages.difference(range(len(hierarchy)))
    if invalid:
        raise ValueError(f"Invalid decoder detach stages: {sorted(invalid)}")

    cloned = []
    for stage, level in enumerate(hierarchy):
        source = level.point if hasattr(level, "point") else level
        feature = source.feat.detach() if stage in detach_stages else source.feat
        data = {
            key: value
            for key, value in source.items()
            if key
            not in {
                "feat",
                "sparse_conv_feat",
                "pooling_parent",
                "unpooling_parent",
            }
        }
        data["feat"] = feature
        if "sparse_conv_feat" in source.keys():
            data["sparse_conv_feat"] = source.sparse_conv_feat.replace_feature(
                feature
            )
        if stage > 0:
            data["pooling_parent"] = cloned[-1]
        cloned.append(Point(data))
    return tuple(cloned)


@MODELS.register_module("LitePT-v1m4")
class LitePTV1M4(LitePTM3):
    """Grid-pooling LitePT with non-mutating encoder/decoder interfaces."""

    def __init__(self, *args, enc_mode=True, **kwargs):
        super().__init__(*args, enc_mode=enc_mode, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.embedding.eval()
            self.enc.eval()
        return self

    def encode(self, data_dict):
        """Run the stem/encoder once and preserve its grid-pooling trace."""
        point = Point(data_dict)
        if self.enc_attn[0]:
            point.serialization(
                order=self.order, shuffle_orders=self.shuffle_orders
            )
        point.sparsify()
        point = self.embedding(point)
        point = self.enc(point)
        hierarchy = build_encoder_hierarchy(point, expected_levels=self.num_stages)
        return point, hierarchy

    def decode_hierarchy(self, hierarchy, detach_stages=()):
        """Decode a clone so VPSD can keep using the encoder hierarchy."""
        if self.enc_mode:
            raise RuntimeError("Encoder-only LitePT-v1m4 has no decoder")
        decoder_levels = _clone_hierarchy_for_decoder(
            hierarchy, detach_stages=detach_stages
        )
        return self.dec(decoder_levels[-1])

    def forward(
        self,
        data_dict,
        return_hierarchy=False,
        decoder_detach_stages=(),
    ):
        point, hierarchy = self.encode(data_dict)
        if not self.enc_mode:
            point = self.decode_hierarchy(
                hierarchy, detach_stages=decoder_detach_stages
            )
        elif decoder_detach_stages:
            raise ValueError(
                "decoder_detach_stages is only valid when enc_mode=False"
            )
        if not return_hierarchy:
            return point
        return point, hierarchy

