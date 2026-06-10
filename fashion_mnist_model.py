"""
fashion_mnist_model.py
----------------------
Custom model definition + custom dataloader for alpha-beta-CROWN.

Loaded via the Customized() primitive in the YAML config:
    model:
      name: Customized("fashion_mnist_model", "fashion_mnist_cnn")
      path: models/fashion_mnist_cnn.pth
    data:
      dataset: Customized("fashion_mnist_model", "fashion_mnist_dataset")
      mean: [0.2860]
      std:  [0.3530]

Follows the signature of cifar10(spec, use_bounds=False) in the official
example custom/custom_model_data.py for this repo version.
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import arguments


def fashion_mnist_cnn():
    """Model architecture. Must match train_model.py exactly so the
    checkpoint loads cleanly."""
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


def fashion_mnist_dataset(spec, use_bounds=False):
    """Custom dataloader for the FashionMNIST test set.

    Mirrors cifar10() in custom_model_data.py: takes a `spec` object,
    reads epsilon from it, and reads mean/std from the config. Returns
    (X, labels, data_max, data_min, eps) where eps is rescaled into the
    normalized input space.
    """
    eps = spec["epsilon"]
    assert eps is not None, "You must specify an epsilon in the config."

    database_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "datasets")
    mean = torch.tensor(arguments.Config["data"]["mean"])
    std = torch.tensor(arguments.Config["data"]["std"])
    normalize = transforms.Normalize(mean=mean, std=std)

    test_data = datasets.FashionMNIST(
        database_path, train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=10000, shuffle=False, num_workers=4)
    X, labels = next(iter(testloader))

    if use_bounds:
        # Element-wise bounds (use only with specification: type = bound).
        absolute_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
        absolute_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
        new_eps = torch.reshape(eps / std, (1, -1, 1, 1))
        data_max = torch.min(X + new_eps, absolute_max)
        data_min = torch.max(X - new_eps, absolute_min)
        ret_eps = None
    else:
        # Single epsilon + clipping bounds. Pixels clipped to [0,1] before
        # normalization -> normalized range [(0-mean)/std, (1-mean)/std].
        data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
        data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
        ret_eps = torch.reshape(eps / std, (1, -1, 1, 1))

    return X, labels, data_max, data_min, ret_eps
