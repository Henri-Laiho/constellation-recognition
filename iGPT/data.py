import torch
from torchvision import datasets
import torchvision.transforms as T

from torch.utils.data import DataLoader, random_split


DATASETS = {
    "constellations": datasets.ImageFolder(root="data", transform=T.ToTensor())
}

def train_transforms(dataset):
    return T.ToTensor()


def test_transforms(dataset):
    return T.ToTensor()


def dataloaders(dataset, batch_size, datapath="data"):
    train_ds = datasets.ImageFolder(root=datapath, transform=T.ToTensor())
    valid_ds = datasets.ImageFolder(root=datapath, transform=T.ToTensor())
    test_ds = datasets.ImageFolder(root=datapath, transform=T.ToTensor())

    # TODO paper uses 90/10 split for every dataset besides ImageNet (96/4)
    train_size = int(0.9 * len(train_ds))

    # reproducable split
    # NOTE: splitting is done twice as datasets have different transforms attributes
    train_ds, _ = random_split(
        train_ds,
        [train_size, len(train_ds) - train_size],
        generator=torch.Generator().manual_seed(0),
    )
    _, valid_ds = random_split(
        valid_ds,
        [train_size, len(valid_ds) - train_size],
        generator=torch.Generator().manual_seed(0),
    )

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=8)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=8)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=8)
    return train_dl, valid_dl, test_dl
