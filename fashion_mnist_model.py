"""
fashion_mnist_model.py
----------------------
Custom model definition + custom dataloader for alpha-beta-CROWN.

This file must be COPIED into the alpha-beta-CROWN repository at:

    alpha-beta-CROWN/complete_verifier/custom/fashion_mnist_model.py

It is then referenced from the YAML config via the Customized() primitive:

    model:
      name: Customized("fashion_mnist_model", "fashion_mnist_cnn")
      path: models/fashion_mnist_cnn.pth
    data:
      dataset: Customized("fashion_mnist_model", "fashion_mnist_dataset")

The structure follows the official tutorial example
complete_verifier/custom/custom_model_data.py (see the repo). If the return
signature of the dataloader changes in a future version, compare against
that file.
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# Must match the normalization used during training (train_model.py).
MEAN = 0.2860
STD = 0.3530


def fashion_mnist_cnn():
    """Model architecture. alpha-beta-CROWN calls this function to build the
    network, then loads the checkpoint given by `model: path` in the config.
    Identical to the definition in train_model.py."""
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),   # 16 x 14 x 14
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 32 x 7 x 7
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )


def fashion_mnist_dataset(eps):
    """Custom dataloader for the FashionMNIST test set.

    Args:
        eps: perturbation radius in *unnormalized* pixel space [0, 1],
             passed from `specification: epsilon` in the YAML config.

    Returns (following the convention of custom_model_data.py):
        X:        normalized test images, shape (N, 1, 28, 28)
        labels:   ground-truth labels, shape (N,)
        data_max: elementwise upper bound of valid (normalized) inputs
        data_min: elementwise lower bound of valid (normalized) inputs
        eps_temp: eps converted into the normalized input space
    """
    assert eps is not None, "specification: epsilon must be set in the config"

    database_path = os.path.join(os.path.dirname(__file__), "..", "datasets")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    test_set = datasets.FashionMNIST(database_path, train=False,
                                     download=True, transform=transform)

    # Load the whole test set into memory; `data: start/end` in the config
    # selects which examples are actually verified.
    loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set),
                                         shuffle=False)
    X, labels = next(iter(loader))

    # Valid pixel range is [0, 1] before normalization. After normalization
    # the valid range becomes [(0 - mean)/std, (1 - mean)/std]. The verifier
    # clips the perturbation ball to this range so that the property is only
    # checked over valid images.
    data_max = torch.tensor((1.0 - MEAN) / STD).reshape(1, -1, 1, 1)
    data_min = torch.tensor((0.0 - MEAN) / STD).reshape(1, -1, 1, 1)

    # eps is given in pixel space; convert it to normalized space.
    eps_temp = torch.tensor(eps / STD).reshape(1, -1, 1, 1)

    return X, labels, data_max, data_min, eps_temp
