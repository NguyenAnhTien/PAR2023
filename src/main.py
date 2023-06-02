"""
@author: Tien Nguyen
@date  : 2023-05-15
"""
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import utils
import constants
from tester import ModelTester
from network import Classifier
from configs import Configurer
from dataset import DatasetHandler
from preprocessor import PreProcessor
from preprocessor import Augmentor

def preprocess(
        configs
    ) -> None:
    """
    @desc:
        - The saved_dir must contains images folders
    """
    dataset_dir = utils.join_path((configs.data_dir, configs.dataset_dir))
    for phase in ['train', 'val']:
        images_dir = eval(f"configs.{phase}_dir")
        label_file = eval(f"configs.{phase}_labels")
        images_dir = utils.join_path((dataset_dir, images_dir))
        label_file = utils.join_path((dataset_dir, label_file))
        preprocessor = PreProcessor(configs, images_dir, label_file, phase)
        preprocessor.run()

def count_invalid_samples(
        configs
    ) -> None:
    dataset_dir = utils.join_path((configs.data_dir, configs.dataset_dir))
    for phase in ['train', 'val']:
        images_dir = eval(f"configs.{phase}_dir")
        label_file = eval(f"configs.{phase}_labels")
        images_dir = utils.join_path((dataset_dir, images_dir))
        label_file = utils.join_path((dataset_dir, label_file))
        preprocessor = PreProcessor(configs, images_dir, label_file, phase)
        invalid_samples = preprocessor.count_invalid_samples()
        print(f"{len(invalid_samples)} invalid samples in {phase} dataset")

def augment(
        configs
    ) -> None:
    augmentor = Augmentor(configs)
    augmentor.run()

def train(
        args,
        configs: Configurer,
    ):
    """
    @desc:
        - train model
    """
    dev_mode = False if args.dev_mode == 'False' else True
    accelerator = 'cuda' if args.gpu > -1 else 'cpu'
    gpu = args.gpu if args.gpu > -1 else 1
    configs.device = accelerator
    model = Classifier(configs)

    logger = WandbLogger(project='PAR2023', job_type='train', tags=['RESNET'])
    trainer = pl.Trainer(max_epochs=configs.epochs, devices=gpu, precision=16,\
                        accelerator=accelerator,\
                        log_every_n_steps=1,\
                         logger=logger,\
                         fast_dev_run=dev_mode)
    trainer.fit(model)

def test_model(
        args,
        configs: Configurer,
    ):
    device = "cuda" if args.gpu > -1 else "cpu"
    checkpoint = args.checkpoint
    report_dir = utils.join_path((configs.log_dir,\
                                                utils.create_report_name()))
    model_tester = ModelTester(device=gpu, configs=configs,\
                                            checkpoint=checkpoint,\
                                                        report_dir=report_dir)
    model_tester.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='False',\
                                                        help='train model')
    parser.add_argument('--test', type=str, default='False',\
                                                        help='test model')
    parser.add_argument('--preprocess', type=str, default='False',\
                                                    help='preprocessing data')
    parser.add_argument('--phase', type=str, default='train',\
                                                        help='train or test')
    parser.add_argument('--dev_mode', type=str, help='DEV or TEST')
    parser.add_argument('--configs', type=str, default='configs.yaml',\
                                                    help='configuration file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu ID')
    parser.add_argument('--count_invalid_samples', type=str, default="False",\
                                                help='count invalid samples')
    parser.add_argument('--augment', type=str, default="False",\
                                                    help='doing augmentation')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file')

    args = parser.parse_args()
    
    configs = Configurer(args.configs)
    if args.train == 'True':
        train(args, configs)
    elif args.test == 'True':
        test_model(args, configs)
    elif args.preprocess == 'True':
        preprocess(configs)
    elif args.count_invalid_samples == 'True':
        count_invalid_samples(configs)
    elif args.augment == 'True':
        augment(configs)
    
