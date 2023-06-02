"""
@author : Tien Nguyen
@date   : 2023-05-21
@desc   :
            - Only load data from preprocess/train
"""
from tqdm import tqdm

from torchvision import transforms

import utils
import constants
from transforms import RandomHorizontalFlip

class Augmentor(object):
    def __init__(
            self, configs
        ):
        self.configs = configs
        self.setup()
        self.define_transforms()
        self.samples = self.load_data()

    def run(
            self
        ):
        for sample in tqdm(self.samples):
            self.augment(sample)

    def augment(
            self, sample_file: str
        ):
        sample = utils.read_pkl(sample_file)
        sample_ = sample.copy()
        basename, file_ext = utils.get_path_basename(sample_file)
        basename = 'aug_' + basename + '.' + file_ext
        saved_file_name = utils.join_path((self.data_dir, basename))
        sample_[constants.IMAGE] = self.transforms(sample_[constants.IMAGE])
        utils.write_pkl(saved_file_name, sample_)

    def define_transforms(
            self
        ) -> None:
        self.transforms = transforms.Compose([
            RandomHorizontalFlip(),
        ])

    def setup(
            self
        ) -> None:
        self.data_dir = utils.join_path((self.configs.preprocess_dir,\
                                                            constants.TRAIN))

    def load_data(
            self
        ) -> list:
        sample_files = utils.list_files(\
                                utils.join_path((self.data_dir, "*.pkl")))
        return sample_files
    