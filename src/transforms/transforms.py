"""
@author : Tien Nguyen
@date   : 2023-05-14
"""
import numpy
import PIL

from torchvision import transforms

import utils

class Resizer(object):
    def __init__(self):
        self.resizer = transforms.Resize(224)

    def __call__(
            self, image
        ):
        resize = self.resizer(image)
        return resize

class CenterCrop(object):
    def __init__(self):
        self.center_crop = transforms.CenterCrop(224)

    def __call__(self, image):
        image = self.center_crop(image)
        return image

class ImageNormalizer(object):
    def __init__(self):
        pass

    def __call__(
            self, 
            image
        ):
        image = numpy.asanyarray(image)
        image = image / 255.0
        return image

class ToTensor(object):
    def __init__(
            self
        ):
        self.to_tensor = transforms.ToTensor()

    def __call__(
            self, image
        ):
        image = image.astype(numpy.float32)
        tensor = self.to_tensor(image)
        return tensor

class Normalize(object):
    def __init__(
            self, mean, std
        ):
        self.normalizer = transforms.Normalize(mean, std)

    def __call__(
            self, image
        ):
        return self.normalizer(image)

class PickleConverter(object):
    def __init__(self):
        pass

    def __call__(self, saved_file, image):
        return utils.write_pkl(saved_file, image)

class RandomHorizontalFlip(object):
    def __init__(
            self
        ) -> None:
        self.transform = transforms.RandomHorizontalFlip(p=1.0)

    def __call__(
            self, image
        ):
        image = self.transform(image)
        return image
