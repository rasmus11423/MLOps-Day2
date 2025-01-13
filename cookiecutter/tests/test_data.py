import sys
import os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from src.code_structure.data import corrupt_mnist
#commentsfad

def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000, "Expected 30000 samples in training set"
    assert len(test) == 5000, "Expected 5000 samples in test set"
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), "Expected each sample to have shape [1, 28, 28]"
            assert y in range(10), "Expected target to be in range 0-9"
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all(), "Expected all targets in training set to be present"
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all() , "Expected all targets in test set to be present"

import torch
from src.code_structure.data import corrupt_mnist

def test_corrupt_mnist():
    train_set, test_set = corrupt_mnist()

    # Check dataset lengths
    assert len(train_set) == 30000
    assert len(test_set) == 5000

    # Check individual data shapes
    for dataset in [train_set, test_set]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)

    # Check unique target labels
    train_targets = torch.unique(train_set.tensors[1])
    test_targets = torch.unique(test_set.tensors[1])
    assert (train_targets == torch.arange(0, 10)).all()
    assert (test_targets == torch.arange(0, 10)).all()

