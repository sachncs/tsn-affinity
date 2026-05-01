"""Observation encoders for different input modalities."""

import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """Encodes observations to transformer embedding dimension.

    Supports two modes:
    - MLP: For 1D vector observations (e.g., CartPole).
    - CNN: For 3D tensor observations (e.g., Atari frames).

    Attributes:
        obs_shape: Shape of observations.
        d_model: Output embedding dimension.
        encoder_type: "mlp" or "cnn".
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        d_model: int,
        encoder_type: str = "mlp",
    ) -> None:
        super().__init__()
        self.obs_shape = obs_shape
        self.d_model = d_model
        self.encoder_type = encoder_type

        if len(obs_shape) == 1:
            self.encoder = self._build_mlp(obs_shape[0])
        elif len(obs_shape) == 3:
            self.encoder = self._build_cnn(obs_shape)
        else:
            raise ValueError(f"Unsupported obs_shape: {obs_shape}")

    def _build_mlp(self, obs_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(obs_dim, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
        )

    def _build_cnn(self, obs_shape: tuple[int, int, int]) -> nn.Module:
        C, _, _ = obs_shape
        return nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, self.d_model),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.encoder(x)  # type: ignore[no-any-return]
