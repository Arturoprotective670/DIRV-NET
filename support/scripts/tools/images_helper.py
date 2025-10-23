import numpy as np
from PIL import Image, ImageDraw
from matplotlib import cm


def array_to_binary_image(image_array):
    # source: https://stackoverflow.com/a/10967471
    return Image.fromarray(np.uint8(cm.binary(image_array) * 255)).convert("L")
