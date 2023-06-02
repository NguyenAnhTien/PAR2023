"""
@author : Tien Nguyen
@date   : 2023-05-14
"""
from tqdm import tqdm

from torchvision import transforms

import utils
import constants
from transforms import Resizer
from transforms import CenterCrop
from transforms import ImageNormalizer
from transforms import Normalize
from transforms import ToTensor
from transforms import PickleConverter

class PreProcessor(object):
    def __init__(
            self, configs, data_sir, label_file, phase
        ) -> None:
        self.phase = phase
        self.configs = configs
        self.data_dir = data_sir
        self.label_file = label_file
        self.setup()
        self.define_transforms()
        self.data = self.load_data()

    def run(
            self
        ) -> None:
        for sample in tqdm(self.data):
            if not self.is_invalid_sample(sample):
                self.preprocess(sample)

    def preprocess(
            self, sample
        ) -> None:
        sample_ = sample.copy()
        image_file = sample[constants.IMAGE_FILE]
        src_img_file = utils.join_path((self.data_dir, image_file))
        saved_sample_file, _ = utils.get_path_basename(image_file)
        saved_sample_file += '.pkl'
        saved_sample_file = utils.join_path((self.saved_images_dir,\
                                                            saved_sample_file))
        image = utils.imread(src_img_file)
        image = self.img_transforms(image)
        sample_[constants.IMAGE] = image
        self.pickle_converter(saved_sample_file, sample_)

    def define_transforms(
            self
        ) -> None:
        self.img_transforms = transforms.Compose([
            Resizer(),
            CenterCrop(),
            ImageNormalizer(),
            ToTensor(),
            Normalize(self.configs.mean, self.configs.std)
        ])

    def load_data(
            self
        ) -> list:
        data = []
        samples = utils.read_txt_file(self.label_file)
        for sample in samples:
            sample = self.parse_label(sample)
            data.append(sample)
        return data

    def parse_label(
            self, sample: str
        ) -> dict:
        segments = sample.split(',')
        image_file = segments[0]
        color_top = int(segments[1])
        color_bottom = int(segments[2])
        gen = int(segments[3])
        bag = int(segments[4])
        hat = int(segments[5])
        return {
            constants.IMAGE_FILE : image_file,
            constants.LABEL      : (color_top, color_bottom, gen, bag, hat)
        }

    def setup(
            self
        ) -> None:
        self.pickle_converter = PickleConverter()
        self.saved_images_dir = utils.join_path((self.configs.preprocess_dir,\
                                                                    self.phase))
        utils.clean_dir(self.saved_images_dir)

    def count_invalid_samples(
            self
        ) -> list:
        invalid_samples = []
        for sample in tqdm(self.data):
            if self.is_invalid_sample(sample):
                invalid_samples.append(sample[constants.IMAGE_FILE])
        return invalid_samples 
    
    def is_invalid_sample(
            self, sample: dict
        ) -> bool:
        labels = sample[constants.LABEL]
        return self.configs.invalid_label in labels
