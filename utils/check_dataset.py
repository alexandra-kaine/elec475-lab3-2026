from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import PetNoseDataset, IMG_SIZE, MEAN, STD


def check_split(split):
    dataset = PetNoseDataset(split=split)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    count = 0

    for images, labels, orig_sizes in tqdm(loader, desc=split):
        assert images.shape[1:] == (3, IMG_SIZE, IMG_SIZE)
        assert labels.shape == (images.shape[0], 2)
        assert orig_sizes.shape == labels.shape

        assert torch.isfinite(images).all()
        assert torch.isfinite(labels).all()

        assert ((labels >= 0) & (labels <= IMG_SIZE)).all(), \
            "Nose coordinate outside image frame"

        assert (orig_sizes > 0).all()

        count += images.shape[0]

    assert count == len(dataset)
    print(f"{split}: {count} samples verified")

    return dataset


def plot_samples(dataset):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)

    fig, axes = plt.subplots(2, 4, figsize=(12, 7))

    for i, ax in enumerate(axes.flat):
        image, label, orig_size = dataset[i]

        # Undo ImageNet normalization
        image = (image * std + mean).clamp(0, 1)
        image = image.permute(1, 2, 0).numpy()

        ax.imshow(image)
        ax.scatter(label[0].item(), label[1].item(),
                   color="red", s=35)

        ax.set_title(f"Training sample {i}")
        ax.axis("off")

    fig.tight_layout()

    Path("plots").mkdir(exist_ok=True)
    fig.savefig("plots/dataset_check.png", dpi=150)
    plt.close(fig)

    print("Saved plots/dataset_check.png")


if __name__ == "__main__":
    train_dataset = check_split("train")
    check_split("test")
    plot_samples(train_dataset)
