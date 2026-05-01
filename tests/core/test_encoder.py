"""Tests for observation encoder."""

import pytest
import torch

from tsn_affinity.core.encoder import ObsEncoder


class TestObsEncoder:
    @pytest.mark.parametrize("encoder_type", ["mlp", "cnn"])
    def test_encoder(self, encoder_type):
        if encoder_type == "mlp":
            encoder = ObsEncoder(obs_shape=(10,), d_model=128, encoder_type="mlp")
            x = torch.randn(4, 10)
            out = encoder(x)
            assert out.shape == (4, 128)
        else:
            encoder = ObsEncoder(obs_shape=(4, 84, 84), d_model=128, encoder_type="cnn")
            x = torch.randn(4, 4, 84, 84)
            out = encoder(x)
            assert out.shape == (4, 128)
