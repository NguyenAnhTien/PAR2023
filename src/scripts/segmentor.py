"""
@author : Tien Nguyen
@date   : 2023-06-13
"""
import os
import glob
import argparse

import pickle

from tqdm import tqdm

from PIL import Image

import torch

from transformers import AutoImageProcessor
from transformers import Mask2FormerForUniversalSegmentation

def read_pkl(
        pkl_file: str
    ):
    """
    @desc:
        - reading a pickle file
    """
    with open(pkl_file, 'rb') as file:
        return pickle.load(file)

def write_pkl(
        pkl_file, data
    ) -> None:
    """
    @desc:
        - store data into a pkl file
    """
    with open(pkl_file, 'wb') as file:
        pickle.dump(data, file)

def read_image(
        image_file: str
    ) -> Image:
    image = Image.open(image_file)
    return image

def load_images(
        images_dir: str
    ) -> list:
    image_files = glob.glob(os.path.join(images_dir, '*.jpg'))
    return image_files

@torch.no_grad()
def segment(
        model,
        processor,
        image_file: str
    ):
    image = read_image(image_file)
    target_sizes=[image.size[::-1]]
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    result = processor.post_process_panoptic_segmentation(outputs,\
                                                target_sizes=target_sizes)[0]
    predicted_panoptic_map = result["segmentation"]
    return predicted_panoptic_map

@torch.no_grad()
def run(
        model,
        processor,
        images_dir
    ):
    phase = images_dir.split()[-1]
    os.makedirs('segments', exist_ok=True)
    os.makedirs(os.path.join('segments', phase), exist_ok=True)
    image_files = load_images(images_dir)
    for image_file in tqdm(image_files[:10]):
        predicted_map = segment(model, processor, image_file)
        predicted_map = predicted_map.cpu().detach().numpy()
        saved_file = image_file.replace("*.jpg", "*.pkl")
        write_pkl(saved_file, predicted_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str,
                                        help='the directory contains images')

    args = parser.parse_args()

    processor = AutoImageProcessor.from_pretrained(\
                                "facebook/mask2former-swin-large-coco-panoptic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(\
                                "facebook/mask2former-swin-large-coco-panoptic")
    run(model, processor, args.images_dir)
