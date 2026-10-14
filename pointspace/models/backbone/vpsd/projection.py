"""Shared native-DINO projector and fixed compact visual readout."""

from math import sqrt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


class NativeDINOProjector(nn.Module):
    """Map a fused point hierarchy into the native DINO feature space."""

    def __init__(self, in_channels, hidden_channels=1024, out_channels=1024):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(int(in_channels)),
            nn.Linear(int(in_channels), int(hidden_channels)),
            nn.GELU(),
            nn.Linear(int(hidden_channels), int(out_channels)),
        )

    def forward(self, feature):
        return self.projector(feature)


class FixedPCAReduction(nn.Module):
    """Apply the globally fitted PCA to normalized native DINO features."""

    def __init__(
        self,
        input_channels=1024,
        output_channels=128,
        pca_path=None,
        require_pca=True,
        output_normalize=True,
        output_scale=None,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.output_normalize = bool(output_normalize)
        self.output_scale = (
            sqrt(self.output_channels)
            if output_scale is None
            else float(output_scale)
        )
        if min(self.input_channels, self.output_channels) <= 0:
            raise ValueError("PCA channel counts must be positive")
        if self.output_channels > self.input_channels:
            raise ValueError("PCA output_channels cannot exceed input_channels")

        if pca_path is None:
            if require_pca:
                raise ValueError("pca_path is required for compact visual output")
            mean = torch.zeros(self.input_channels, dtype=torch.float32)
            components = torch.eye(
                self.output_channels, self.input_channels, dtype=torch.float32
            )
        else:
            mean, components = self._load_pca(pca_path)
        self.register_buffer("mean", mean.contiguous(), persistent=True)
        self.register_buffer("components", components.contiguous(), persistent=True)

    def _load_pca(self, pca_path):
        path = Path(pca_path)
        if not path.is_file():
            raise FileNotFoundError(f"PCA model does not exist: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if not {"mean", "components"}.issubset(keys):
                raise ValueError(
                    "PCA model must contain 'mean' and 'components' tensors"
                )
            mean = handle.get_tensor("mean").float()
            components = handle.get_tensor("components").float()
            metadata = dict(handle.metadata() or {})
        if tuple(mean.shape) != (self.input_channels,):
            raise ValueError(
                f"PCA mean shape {tuple(mean.shape)} does not match "
                f"({self.input_channels},)"
            )
        expected = (self.output_channels, self.input_channels)
        if tuple(components.shape) != expected:
            raise ValueError(
                f"PCA components shape {tuple(components.shape)} does not match "
                f"{expected}"
            )
        if metadata.get("normalize_input", "").lower() != "true":
            raise ValueError("Visual PCA requires normalize_input=True")
        if metadata.get("whiten", "").lower() != "false":
            raise ValueError("Visual PCA requires whiten=False")
        return mean, components

    def project_normalized(self, normalized_feature, output_dtype=None):
        with torch.autocast(
            device_type=normalized_feature.device.type, enabled=False
        ):
            feature = normalized_feature.float() - self.mean
            reduced = F.linear(feature, self.components)
            if self.output_normalize:
                reduced = F.normalize(reduced, dim=-1, eps=1e-12)
                reduced = reduced * self.output_scale
        return reduced.to(
            normalized_feature.dtype if output_dtype is None else output_dtype
        )

    def forward(self, feature):
        output_dtype = feature.dtype
        with torch.autocast(device_type=feature.device.type, enabled=False):
            normalized = F.normalize(feature.float(), dim=-1, eps=1e-12)
        return self.project_normalized(normalized, output_dtype=output_dtype)
