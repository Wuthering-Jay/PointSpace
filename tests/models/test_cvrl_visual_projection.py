import numpy as np
import torch
from safetensors.numpy import save_file

from pointspace.models.backbone.vpsd.projection import FixedPCAReduction


def test_fixed_pca_visual_readout_and_native_dimension(tmp_path):
    rng = np.random.default_rng(42)
    components = rng.normal(size=(4, 8)).astype(np.float32)
    components /= np.linalg.norm(components, axis=1, keepdims=True)
    mean = rng.normal(size=8).astype(np.float32)
    path = tmp_path / "pca.safetensors"
    save_file(
        {"mean": mean, "components": components},
        str(path),
        {"normalize_input": "true", "whiten": "false"},
    )
    projection = FixedPCAReduction(
        input_channels=8,
        output_channels=4,
        pca_path=path,
        output_normalize=True,
        output_scale=2.0,
    )
    native = torch.randn(7, 8)
    compact = projection(native)
    assert native.shape == (7, 8)
    assert compact.shape == (7, 4)
    assert torch.allclose(
        torch.linalg.vector_norm(compact.float(), dim=1),
        torch.full((7,), 2.0),
        atol=1e-5,
        rtol=1e-5,
    )
