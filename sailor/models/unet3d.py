"""Compact 3D U-Net shared by C0, C1, and the G4 constant-Δt control."""

from __future__ import annotations

from typing import Any

from sailor.constants import BASELINE_IN_CHANNELS
from sailor.errors import StopProtocolError


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise StopProtocolError(
            "PyTorch is not installed.",
            "C0 and C1 cannot be trained or scored.",
            "Install the pinned Phase-3 torch extra in the runtime that will train.",
        ) from exc
    return torch, nn


def build_unet(*, in_channels: int = BASELINE_IN_CHANNELS, base_channels: int = 8):
    torch, nn = require_torch()

    class DoubleConv(nn.Module):
        def __init__(self, start: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv3d(start, out_channels, 3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, inputs):  # type: ignore[no-untyped-def]
            return self.block(inputs)

    class BaselineUNet3D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc1 = DoubleConv(in_channels, base_channels)
            self.enc2 = DoubleConv(base_channels, base_channels * 2)
            self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
            self.pool = nn.MaxPool3d(2)
            self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)
            self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
            self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
            self.dec1 = DoubleConv(base_channels * 2, base_channels)
            self.head = nn.Conv3d(base_channels, 1, 1)
            self.contract = {
                "in_channels": in_channels,
                "base_channels": base_channels,
                "output_channels": 1,
            }

        def forward(self, inputs):  # type: ignore[no-untyped-def]
            if inputs.ndim != 5 or inputs.shape[1] != in_channels:
                raise ValueError(
                    f"Baseline U-Net expected (B,{in_channels},D,H,W); got {tuple(inputs.shape)}."
                )
            skip1 = self.enc1(inputs)
            skip2 = self.enc2(self.pool(skip1))
            center = self.enc3(self.pool(skip2))
            up2 = self.up2(center)
            up2 = _crop_or_pad(up2, skip2.shape[2:])
            decoded2 = self.dec2(torch.cat([up2, skip2], dim=1))
            up1 = self.up1(decoded2)
            up1 = _crop_or_pad(up1, skip1.shape[2:])
            decoded1 = self.dec1(torch.cat([up1, skip1], dim=1))
            return self.head(decoded1)

        def describe(self) -> dict[str, Any]:
            n_params = sum(parameter.numel() for parameter in self.parameters())
            return {**self.contract, "n_parameters": int(n_params)}

    def _crop_or_pad(volume, spatial):  # type: ignore[no-untyped-def]
        current = tuple(volume.shape[2:])
        if current == tuple(spatial):
            return volume
        slices = []
        pads = []
        for size, target in zip(current, spatial):
            if size >= target:
                start = (size - target) // 2
                slices.append(slice(start, start + target))
                pads.append((0, 0))
            else:
                slices.append(slice(0, size))
                extra = target - size
                pads.append((extra // 2, extra - extra // 2))
        cropped = volume[(slice(None), slice(None), *slices)]
        pad = []
        for before, after in reversed(pads):
            pad.extend([before, after])
        if any(pads):
            cropped = torch.nn.functional.pad(cropped, pad)
        return cropped

    return BaselineUNet3D()
