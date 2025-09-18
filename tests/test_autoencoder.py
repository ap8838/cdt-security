import torch

from src.models.autoencoder import Autoencoder


def test_autoencoder_forward():
    # simple input: batch of 4 samples, 10 features each
    x = torch.randn(4, 10)
    model = Autoencoder(input_dim=10)

    out = model(x)

    # Output should have same shape as input
    assert out.shape == x.shape
