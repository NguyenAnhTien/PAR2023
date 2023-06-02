"""
@author: Tien Nguyen
@date  : 2023-05-15
"""

from PIL import Image

def imread(
        image_file: str
    ) -> Image:
    """
    """
    image = Image.open(image_file)
    return image
