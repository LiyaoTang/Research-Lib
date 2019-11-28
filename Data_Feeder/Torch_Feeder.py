#!/usr/bin/env python
# coding: utf-8
"""
module: pipeline to solve data-related problem for neural net built by py-torch (using torch.utils.data.Dataset)
    note: torch conv default input: [N, C, H, W], N=batch size, C=channel, H/W = high/width
"""

import torch
import numpy as np

from torch.utils.data import Dataset, IterableDataset
from . import Data_Feeder as feeder

class Torch_Feeder(Dataset):
    """
    wrapper from feeder to torch data.Dataset
    """

    def __init__(self, feeder_list):
        super(Torch_Feeder, self).__init__()
        self.feeder_list = feeder_list

        data_num_list = [len(f.data_ref) for f in self.feeder_list]
        self.data_num = sum(data_num_list)
        self.feeder_split = np.array([sum(data_num_list[:i]) for i in range(len(self.feeder_list))])

    def __getitem__(self, index):
        # assume data_ref in each feeder is concat according to self.feeder_list
        idx_arr = self.feeder_split - index  # conver index into idx in data_ref of each feeder
        feeder_idx = np.where(idx_arr >= 0)[0][0]  # find the feeder contain the desired index
        cur_idx = idx_arr[feeder_idx]
        cur_feeder = self.feeder_list[feeder]
        cur_ref = cur_feeder.data_ref[idx]
        return cur_feeder._get_input_label_pair(cur_ref)

    def __len__(self):
        return self.data_num

    def reset():
        for f in self.feeder_list:
            f.reset()