import os
import random
import torch
import numpy as np
import cupy as cp
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from PIL import Image
from torch.utils.data import Dataset


class DataIter(Dataset):
    def __init__(self, data_holder, config, dataset_name, gpu):
        super().__init__()
        self.gpu = gpu
        self.data_dir = data_holder.data_dir
        self.config = config
        self.num_repeats = config['num_repeats'] if dataset_name.lower() == 'train' else 1

        self.dataset = data_holder.get_dataset(dataset_name)
        self.indices = list(self.dataset['data'])
        self.init_worker = True

    def __getitem__(self, item):
        if self.init_worker:
            self.init_worker = False
            cp.cuda.Device(self.gpu).use()
            np.random.seed(random.randint(0, 2**32))
            cp.random.seed(random.randint(0, 2**32))

        data_id = self.indices[item % self.dataset['size']]
        data = self.dataset['data'][data_id]

        noisy = np.load(os.path.join(self.data_dir, data['noisy']))[np.newaxis,:,:]
        clean = np.load(os.path.join(self.data_dir, data['clean']))[np.newaxis, :, :]
        noisy = torch.as_tensor(noisy, device=self.gpu)
        clean = torch.as_tensor(clean, device=self.gpu)
        return noisy, clean

    def __len__(self):
        return self.dataset['size'] * self.num_repeats if self.dataset['train'] else self.dataset['size']
