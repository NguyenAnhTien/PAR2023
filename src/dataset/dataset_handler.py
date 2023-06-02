"""
@author : KHANH THUONG TRAN
@date   : 2023-05-08
@update : Tien Nguyen
"""

from torch.utils.data import Dataset

import utils
import constants

class DatasetHandler(Dataset):
    def __init__(
            self, data_dir: str
        ):
        self.data_dir = data_dir
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_file = self.data[idx]
        sample = utils.read_pkl(sample_file)
        sample[constants.LABEL] = self.transform_labels(sample[constants.LABEL])
        return sample
    
    def transform_labels(
            self, labels: tuple
        ):
        color_top = int(labels[0])
        color_bottom = int(labels[1])
        gen = int(labels[2])
        bag = int(labels[3])
        hat = int(labels[4])
        return (color_top, color_bottom, gen, bag, hat)

    def load_data(
            self
        ) -> list:
        sample_files = utils.list_files(\
                                utils.join_path((self.data_dir, "*.pkl")))
        return sample_files
