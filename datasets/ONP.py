"""
Online News Popularity (ONP)
Description: Contains 2-years data, we split it into 24 domains by month
URL: https://github.com/krishnakartik1/onlineNewsPopularity
Author: QIN Tiexin
"""

# -*- coding: utf-8 -*-
import os
import pdb

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset

from datasets.dataset import MultipleDomainDataset
from engine.configs import Datasets


class MultipleEnvironmentONP(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape, num_classes):
        super().__init__()
        num_domains = 24
        self.Environments = np.arange(num_domains)

        self.root = root
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.normalize = False

        data_dict = self.load_data_dict()
        all_time_stamps = list(data_dict.keys())

        months = [date[:7] for date in all_time_stamps]
        months = list(set(months))
        months = sorted(months)

        self.datasets = []

        for domain_idx, month in enumerate(months):
            cur_domain_data = []
            for time_stamp in all_time_stamps:
                if str(time_stamp).startswith(str(month)):
                    cur_domain_data.extend(data_dict[str(time_stamp)])

            cur_domain_data = np.stack(cur_domain_data)
            data = cur_domain_data[:, :-1].astype(float)
            labels = cur_domain_data[:, -1].astype(int)
            self.datasets.append(dataset_transform(data, labels))


    def load_data_dict(self):
        data = pd.read_csv(self.root)

        data_dict = {}

        for index, row in data.iterrows():
            time_stamp = row['time_stamp']
            cur_data = row.drop(['time_stamp']).to_numpy()

            if time_stamp not in data_dict:
                data_dict[time_stamp] = [cur_data]
            else:
                data_dict[time_stamp].append(cur_data)

        return data_dict  # 720 date


@Datasets.register('onp')
class ONP(MultipleEnvironmentONP):
    def __init__(self, root, input_shape, num_classes):
        num_domains = 24
        environments = list(np.arange(num_domains))

        super(ONP, self).__init__(root, environments, self.process_dataset, input_shape, num_classes)

    def process_dataset(self, data, labels):
        x = torch.tensor(data).float()
        y = torch.tensor(labels).long()
        return TensorDataset(x, y)


if __name__ == '__main__':
    file_root = '../data/processed_ONP.csv'
    input_shape = [1, 58]
    num_classes = 2
    ds = ONP(file_root, input_shape, num_classes)