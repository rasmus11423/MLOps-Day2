import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

import torch
from src.code_structure.model import MyAwesomeModel
from src.code_structure.data import corrupt_mnist

def test_training_convergence():
    """
    Test that the training loss decreases over the first few iterations.
    """
    # Set up a small dataset and model for testing
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)

    model = MyAwesomeModel()
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial_loss = None
    for i, (img, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        y_pred = model(img)
        loss = loss_fn(y_pred, target)
        loss.backward()
        optimizer.step()

        if i == 0:
            initial_loss = loss.item()
        if i == 5:  # Check the loss after 5 iterations
            assert loss.item() < initial_loss, "Training loss did not decrease over iterations"
            break
