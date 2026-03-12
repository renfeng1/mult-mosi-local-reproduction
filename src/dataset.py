import os
import pickle

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data="mosei_senti", split_type="train", if_align=False):
        super().__init__()
        dataset_path = os.path.join(
            dataset_path, data + "_data.pkl" if if_align else data + "_data_noalign.pkl"
        )
        with open(dataset_path, "rb") as handle:
            dataset = pickle.load(handle)

        self.vision = torch.from_numpy(dataset[split_type]["vision"].astype(np.float32)).cpu()
        self.text = torch.from_numpy(dataset[split_type]["text"].astype(np.float32)).cpu()

        audio = dataset[split_type]["audio"].astype(np.float32)
        audio[audio == -np.inf] = 0
        self.audio = torch.from_numpy(audio).cpu()

        self.labels = torch.from_numpy(dataset[split_type]["labels"].astype(np.float32)).cpu()
        self.meta = dataset[split_type]["id"] if "id" in dataset[split_type].keys() else None
        self.data = data
        self.n_modalities = 3

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def get_lbl_info(self):
        return self.labels.shape[1], self.labels.shape[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]

        if self.meta is None:
            meta = (0, 0, 0)
        else:
            meta = (self.meta[index][0], self.meta[index][1], self.meta[index][2])

        if self.data == "mosi" and self.meta is not None:
            meta = tuple(
                item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else item for item in meta
            )
        if self.data == "iemocap":
            Y = torch.argmax(Y, dim=-1)
        return X, Y, meta
