import sys, os

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

from lfm_dataset.cub import CUB2002011Dataset
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from libs.autoencoder import get_model
import argparse
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms

torch.manual_seed(0)
np.random.seed(0)

def main(resolution=256):

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = CUB2002011Dataset(root_dir="/home/longteng/datasets/cub/CUB_200_2011", transform=transform)

    train_dataset_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    model = get_model("assets/stable-diffusion/autoencoder_kl.pth")
    model = nn.DataParallel(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # features = []
    # labels = []

    idx = 0
    for batch in tqdm(train_dataset_loader):
        img, segmask, attr = batch

        img = img.to(device)
        moments = model(img, fn="encode_moments")
        moments = moments.detach().cpu().numpy()

        segmask = segmask.detach().cpu().numpy()

        save_dir = Path(f'assets/datasets/cub{resolution}_features_with_supervision')
        save_dir.mkdir(parents=True, exist_ok=True)

        for _moment, _segmask, _attr in zip(moments, segmask, attr):

            np.save(
                f"assets/datasets/cub{resolution}_features_with_supervision/{idx}.npy",
                (_moment, _segmask, _attr),
            )
            idx += 1

    print(f"save {idx} files")

    # features = np.concatenate(features, axis=0)
    # labels = np.concatenate(labels, axis=0)
    # print(f'features.shape={features.shape}')
    # print(f'labels.shape={labels.shape}')
    # np.save(f'imagenet{resolution}_features.npy', features)
    # np.save(f'imagenet{resolution}_labels.npy', labels)


if __name__ == "__main__":
    main()