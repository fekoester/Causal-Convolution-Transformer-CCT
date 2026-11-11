import numpy as np
import torch
from torch.utils.data import Dataset


class LMDataset(Dataset):
    """
    Language-modelling windows from a pre-tokenized uint16 .bin file, memory-mapped (the file is never
    loaded into RAM as a whole). Window i covers tokens [i*seq_length, (i+1)*seq_length); inputs are its
    first seq_length-1 tokens and targets the last seq_length-1.
    """

    def __init__(self, bin_path: str, seq_length: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_length = int(seq_length)
        self.n_seq = len(self.data) // self.seq_length
        if self.n_seq == 0:
            raise ValueError(f"{bin_path} has {len(self.data)} tokens, fewer than one window of {seq_length}")

    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        s = idx * self.seq_length
        seq = torch.from_numpy(np.asarray(self.data[s: s + self.seq_length]).astype(np.int64))
        return seq[:-1], seq[1:]
