"""
@author : Tien Nguyen
@date   : 2023-06-01
"""
from tqdm import tqdm

import torch

import utils
import constants
from configs import Configurer
from network import Classifier
from dataset import DatasetHandler

class ModelTester(object):
    def __init__(
            self,
            device: str,
            configs: Configurer,
            checkpoint: str, 
            report_dir: str
        ) -> None:
        self.device = device
        self.configs = configs
        self.checkpoint = checkpoint
        self.report_dir = report_dir
        self.setup()

    def run(
            self
        ) -> None:
        train_dict = self.report(self.model.train_data_handler, constants.TRAIN)
        val_dict = self.report(self.model.val_data_handler, constants.VAL)
        result = {
            constants.VAL : val_dict,
            constants.TRAIN : train_dict,
            constants.CHECKPOINT : self.checkpoint,
        }
        utils.write_json(result, utils.join_path((self.report_dir,\
                                                        constants.REPORT_FILE)))

    def report(
            self,
            phase: str,
            dataset_handler: DatasetHandler,
        ) -> dict:
        result_df = self.predict(dataset_handler, phase)
        return {
            "mA" : None
        }
    
    @torch.no_grad()
    def predict(
            self,
            dataset_handler: DatasetHandler
        ) -> tuple:
        result_df = {
            constants.FILE_ID       : [],
            constants.PREDICT       : [],
            constants.MEAN_ACC      : [],
            constants.LABEL.upper() : []
        }
        for sample in tqdm(dataset_handler):
            if 'aug' in sample[constants.IMAGE_FILE]:
                continue
            image_file = sample[constants.IMAGE_FILE]
            file_id = utils.get_path_basename(image_file)
            image = sample[constants.IMAGE]
            label = sample[constants.LABEL]
            image = image.to(self.device)
            output = self.model.predict(image)
            result_df[constants.FILE_ID].append(file_id)
            result_df[constants.PREDICT].append(output)
            result_df[constants.LABEL].append(label)
            import ipdb
            ipdb.set_trace()
        return result_df

    def load_checkpoint(
            self
        ):
        """
        @desc:
            -) load model from a checkpoint file.
        """
        model = Classifier.load_from_checkpoint(self.checkpoint)
        model.to(self.device)
        model.eval()
        return model
    
    def setup(
            self
        ) -> None:
        self.model = self.load_checkpoint()
        image_dir = utils.join_path((self.configs.preprocess_dir, constants.VAL))
        self.dataset_handler = DatasetHandler(image_dir)
