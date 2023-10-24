

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from lfm_dataset.cub import CUB2002011Dataset

class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = tuple(self.dataset[item][:-1])  # remove label
        if len(data) == 1:
            data = data[0]
        return data


class DatasetFactory(object):
    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.0)
        v.clamp_(0.0, 1.0)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError

class CommonFeatureDataset(Dataset):
    def __init__(self, path, np_num):
        super().__init__()
        self.path = path
        self.np_num = np_num
        print("CommonFeatureDataset load", path)

    def __len__(self):
        return self.np_num

    def __getitem__(self, idx):
        path = os.path.join(self.path, f"{idx}.npy")
        z, attr = np.load(path, allow_pickle=True)

        return z, attr

class CommonFeatureDataset_CM_Conditional(Dataset):
    def __init__(self, path, np_num):
        super().__init__()
        self.path = path
        self.np_num = np_num
        print("CommonFeatureDataset load", path)

    def __len__(self):
        return self.np_num

    def __getitem__(self, idx):
        path = os.path.join(self.path, f"{idx}.npy")
        data = np.load(path, allow_pickle=True)
        if len(data) == 3: # (8, 32, 32), (1, 256, 256) for CUB256
            z, mask, attr = data
            return z, mask
        elif len(data) == 2:
            z, mask = data # (8, 32, 32), (1, 256, 256) for CM256
            return z, mask
        else:
            raise ValueError

class CM256Features_Cond(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print("Prepare dataset...")
        self.train = CommonFeatureDataset_CM_Conditional(path, np_num=30_001 - 1)
        print("Prepare dataset ok")
        self.K = None

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz"  # temporaly, TODO

    def sample_label(self, n_samples, device):
        raise NotImplementedError
        return torch.randint(0, 1000, (n_samples,), device=device)
    
class CUB256Features(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print("Prepare dataset...")
        self.train = CommonFeatureDataset_CM_Conditional(path, np_num=11788)
        print("Prepare dataset ok")
        self.K = None

        if cfg:  # classifier free guidance
            raise NotImplementedError
            assert p_uncond is not None
            print(
                f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}"
            )
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz"  # temporaly, TODO

    def sample_label(self, n_samples, device):
        raise NotImplementedError
        return torch.randint(0, 1000, (n_samples,), device=device)


class CM256Features(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print("Prepare dataset...")
        self.train = CommonFeatureDataset(path, np_num=30_001 - 1)
        print("Prepare dataset ok")
        self.K = None

        if cfg:  # classifier free guidance
            raise NotImplementedError
            assert p_uncond is not None
            print(
                f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}"
            )
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz"  # temporaly, TODO

    def sample_label(self, n_samples, device):
        raise NotImplementedError
        return torch.randint(0, 1000, (n_samples,), device=device)



class Churches256Features(
    DatasetFactory
):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print("Prepare dataset...")
        np_num = 242455 - 1
        self.train = CommonFeatureDataset(path, np_num=np_num)
        print("Prepare dataset ok")
        self.K = None

        if cfg:  # classifier free guidance
            raise NotImplementedError
            assert p_uncond is not None
            print(
                f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}"
            )
            self.train = CFGDataset(self.train, p_uncond, self.K)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz"  # temporaly, TODO

    def sample_label(self, n_samples, device):
        raise NotImplementedError
        return torch.randint(0, 1000, (n_samples,), device=device)


def get_dataset(name, **kwargs):

    if name == "celebamask256_features":
        return CM256Features(**kwargs)
    elif name == "celebamask256_features_cond":
        return CM256Features_Cond(**kwargs)
    elif name == "cub256_features":
        return CUB256Features(**kwargs)
    elif name == "churches256_features":
        return Churches256Features(**kwargs)
