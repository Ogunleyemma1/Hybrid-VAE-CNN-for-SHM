# src/models/cnn_model.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """
    2D CNN classifier for window-level discrimination.

    Expected input shape:
        x: (B, 1, T, C)
            B: batch size
            T: sequence length (window length)
            C: number of channels/features

    Output:
        logits: (B, num_classes)

    Notes:
    - Uses Conv2D over (time, features) to learn joint temporal-feature patterns.
    - GroupNorm is used for stable training independent of batch size.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_width: int = 32,
        dropout: float = 0.4,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or num_classes <= 0 or base_width <= 0:
            raise ValueError("in_channels, num_classes, and base_width must be positive integers.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if gn_groups <= 0:
            raise ValueError("gn_groups must be positive.")

        def conv_block(cin: int, cout: int, kt: int, kf: int, pt: int, pf: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=(kt, kf), padding=(pt, pf), bias=False),
                nn.GroupNorm(num_groups=min(gn_groups, cout), num_channels=cout),
                nn.SiLU(inplace=True),
            )

        w1, w2, w3, w4 = base_width, base_width * 2, base_width * 4, base_width * 8

        self.features = nn.Sequential(
            conv_block(in_channels, w1, kt=7, kf=3, pt=3, pf=1),
            nn.MaxPool2d(kernel_size=(2, 1)),  # downsample time only

            conv_block(w1, w2, kt=5, kf=3, pt=2, pf=1),
            nn.MaxPool2d(kernel_size=(2, 1)),

            conv_block(w2, w3, kt=5, kf=3, pt=2, pf=1),
            nn.MaxPool2d(kernel_size=(2, 1)),

            conv_block(w3, w4, kt=3, kf=3, pt=1, pf=1),
            nn.AdaptiveAvgPool2d((1, 1)),      # (B, w4, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w4, w2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(w2, num_classes),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming initialization for conv/linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B,1,T,C), got {tuple(x.shape)}.")
        if x.size(1) != 1:
            raise ValueError(f"Expected channel dimension x[:,1,...] == 1, got {x.size(1)}.")

        x = self.features(x)
        logits = self.classifier(x)
        return logits
