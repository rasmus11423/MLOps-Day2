import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.code_structure.model import MyAwesomeModel
import torch
import pytest


@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)


# def test_error_on_wrong_shape():
#     model = MyAwesomeModel()
#     with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
#         model(torch.randn(1,2,3))
#     with pytest.raises(ValueError, match='Expected each sample to have shape [1, 28, 28]'):
#         model(torch.randn(1,1,28,29))

def test_forward_pass():
    model = MyAwesomeModel()
    x = torch.randn(8, 1, 28, 28)  # Batch of 8, 1 channel, 28x28
    y = model(x)

    # Check output shape
    assert y.shape == (8, 10), "Output should have shape [batch_size, 10]"

    # Check output is a tensor
    assert isinstance(y, torch.Tensor)
