"""
@author : Tien Nguyen
@date   : 2023-06-28
"""
import os
import random
import numpy
import cv2
import csv
import argparse

import tqdm

import utils
from configs import Configurer
from network import Classifier
from preprocessor import PreProcessor


def read_csv(
        args: argparse.Namespace,
    ) -> tuple:
    with open(args.data, mode='r') as csv_file:
        gt = csv.reader(csv_file, delimiter=',')
        gt_num = 0
        gt_dict = {}
        for row in gt:
            gt_dict.update({row[0]: int(round(float(row[1])))})
            gt_num += 1
    return gt_dict, gt_num

def load_model():
    checkpoint = os.path.join('weights', 'model-epoch=01-val_loss=3.26.ckpt')
    model = Classifier.load_from_checkpoint(checkpoint)
    model.eval()
    return model
    
def inference(
        model,
        preprocessor: PreProcessor,
        gt_dict: dict,
        args: argparse.Namespace,
    ) -> None:
    with open(args.results, 'w', newline='') as res_file:
        writer = csv.writer(res_file)
        for image_file in tqdm.tqdm(gt_dict.keys()):
            image = utils.imread(os.path.join(args.images, image_file))
            image = preprocessor(image)
            pred = model.predict(image).cpu().detach().tolist()[0]
            writer.writerow([image_file, pred[0], pred[1],\
                                                pred[2], pred[3], pred[4]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--data", type=str, default='foo_test.csv', help="Dataset labels")
    parser.add_argument("--images", type=str, default='foo_test/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results.csv', help="CSV file of the results")
    
    args = parser.parse_args()

    gt_dict, gt_num = read_csv(args)
    configs = Configurer('configs.yaml')
    preprocessor = PreProcessor(configs, args.images, args.data, 'val').img_transforms
    model = load_model()
    inference(model, preprocessor, gt_dict, args)
