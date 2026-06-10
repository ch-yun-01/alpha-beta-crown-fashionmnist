"""
train_model.py
--------------
Trains a small CNN on FashionMNIST and saves the checkpoint that will be
verified by alpha-beta-CROWN.

FashionMNIST is NOT one of alpha-beta-CROWN's built-in models/datasets
(the models directory only contains MNIST / CIFAR-10 / CIFAR-100 /
TinyImageNet models), so this satisfies the "external model and dataset"
requirement of the assignment.

The architecture is intentionally small (2 conv + 2 fc, ReLU only) so that
complete verification with branch-and-bound finishes quickly.

Usage:
    python train_model.py            # trains and saves fashion_mnist_cnn.pth
    python train_model.py --epochs 10
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Normalization constants for FashionMNIST.
# IMPORTANT: these must match the values used in the custom dataloader
# (fashion_mnist_model.py) and implicitly in the YAML config, otherwise the
# verified region would not correspond to a true L_inf ball in pixel space.
MEAN = 0.2860
STD = 0.3530


def fashion_mnist_cnn():
    """Small ReLU CNN for FashionMNIST (1x28x28 -> 10 classes).

    Same function is duplicated in fashion_mnist_model.py, which is the file
    that alpha-beta-CROWN loads via the Customized() primitive. Keep the two
    definitions identical.
    """
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="fashion_mnist_cnn.pth")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    train_set = datasets.FashionMNIST("./data", train=True, download=True,
                                      transform=transform)
    test_set = datasets.FashionMNIST("./data", train=False, download=True,
                                     transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=512, shuffle=False, num_workers=2)

    model = fashion_mnist_cnn().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)

        # Evaluate clean accuracy. Clean accuracy matters for verification:
        # misclassified samples are trivially "falsified", so we want a model
        # with high clean accuracy on the first N test samples.
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
        acc = correct / len(test_set)
        print(f"epoch {epoch + 1}/{args.epochs}  "
              f"train_loss={total_loss / len(train_set):.4f}  "
              f"test_acc={acc:.4f}")

    # Save only the state_dict; alpha-beta-CROWN re-creates the architecture
    # from the Customized() model definition and loads this checkpoint.
    torch.save(model.state_dict(), args.out)
    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
