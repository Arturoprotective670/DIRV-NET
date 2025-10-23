from array import array
from PIL import Image

# from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
#                                 denoise_wavelet, estimate_sigma)
import numpy as np

from tools.trim_image import auto_trim_array


def pre_process_image(data: array) -> array:
    data = auto_trim_array(data)

    if len(data) > 0:
        # resize the smallest dimension, to 84 pixels
        image_dim = 84
        image = Image.fromarray(data)
        if data.shape[0] < data.shape[1]:
            size = (round(image_dim * data.shape[1] / data.shape[0]), image_dim)
        else:
            size = (image_dim, round(image_dim * data.shape[0] / data.shape[1]))

        image = image.resize(size, Image.ANTIALIAS)
        data = np.asarray(image)

    return data
