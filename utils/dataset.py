from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMG_SIZE = 227

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class PetNoseDataset(Dataset):

    def __init__(self, root: str = "data/oxford-iiit-pet-noses", split: str = "train"):
        super(PetNoseDataset, self).__init__()

        self.img_dir = Path(root) / "images-original" / "images"
        label_file = Path(root) / f"{split}_noses.txt"

        self.samples = []

        with open(label_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                name, coords = line.strip().split(",", 1)
                coords = coords.strip().strip('"').strip("()")
                u, v = map(float, coords.split(","))

                self.samples.append((name, u, v))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):

        name, u, v = self.samples[idx]

        img = Image.open(self.img_dir / name).convert("RGB")

        w, h = img.size

        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        u = u * IMG_SIZE / w
        v = v * IMG_SIZE / h

        image = TF.to_tensor(img)
        image = TF.normalize(image, MEAN, STD)

        label = torch.tensor([u, v], dtype=torch.float32)
        orig_size = torch.tensor([w, h], dtype=torch.float32)

        return image, label, orig_size


def denormalize(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


if __name__ == "__main__":
    print("Testing PetNoseDataset:")

    for split in ["train", "test"]:
        ds = PetNoseDataset(split=split)
        image, label, orig_size = ds[0]

        print(f"  {split:5s}: {len(ds):,} samples | image {tuple(image.shape)} "
              f"| label {label.tolist()} | orig_size {orig_size.tolist()}")
