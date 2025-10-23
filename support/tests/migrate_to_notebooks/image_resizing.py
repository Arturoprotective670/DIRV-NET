"""
This script is to check if the implemented 2D and 3D resizing working correctly.
"""

# below is needed to help python references lookups when this
# script is run separately, source https://stackoverflow.com/a/4383597
import math
import sys

STARTUP_DIR = (
    "/mnt/asgard2/code/nadim/case2d_code_copy/"  # change this absolute path to yours
)
sys.path.insert(1, STARTUP_DIR)


from source.model.tools.operators.resizing_operator import (
    ResizingOperator,
    ResizingMethods,
)
from PIL import Image
import numpy as np
import tensorflow as tf


def save(tensor, name, multichannel, is_3d):
    img = tensor[0, ...].numpy()

    if not multichannel:
        img = img[..., 0]
        if is_3d:
            img = img[..., img.shape[-1] // 2]

    if multichannel:
        img = Image.fromarray(np.uint8(img), "RGB")
    else:
        img = Image.fromarray(np.uint8(img)).convert("L")

    img.save(name + ".jpg")


def check_3D():
    FILE_PATH = "/mnt/asgard2/code/nadim/case2d_code_copy/data/h5_experiment/origin/4DCT_case1.h5"
    # according to Isotropic Total Variation Regularization of
    # Displacements in Parametric Image Registration
    # Valeriy Vishnevskiy, Christine Tanner, Orcun Goksel et. al. paper
    MAX_CLIP = 1200
    MIN_CLIP = 50
    from support.scripts.file_formats.h5_format import read_h5

    res = read_h5(FILE_PATH)
    first_pase = res[..., 0]

    img = np.clip(first_pase, MIN_CLIP, MAX_CLIP)
    img = (img - MIN_CLIP) / (MAX_CLIP - MIN_CLIP)
    img = img[tf.newaxis, :, :, :, tf.newaxis]

    img = tf.convert_to_tensor(img, dtype=tf.float32)
    br = ResizingOperator(tf.shape(img), method=ResizingMethods.LINEAR_SMOOTHED)

    down_img = br.resize_down(img, [2, 2, 2])
    up_img = br.resize_up(img, [2, 2, 2])

    save(img * 255, "original_3d", False, True)
    save(down_img * 255, "scaled_down_3d", False, True)
    save(up_img * 255, "scaled_up_3d", False, True)


def check_2D_mono():
    IMAGE_PATH = "/mnt/asgard2/code/nadim/case2d_code_copy/support.scripts/5af89c8.jpg"
    img = np.array(Image.open(IMAGE_PATH))
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = img[tf.newaxis, :, :, tf.newaxis]

    br = ResizingOperator(tf.shape(img), method=ResizingMethods.B_SPLINES)
    down_img = br.resize_down(img, [2, 2])
    up_img = br.resize_down(img, [2, 2])

    save(img, "original_2d_mono", False, False)
    save(down_img, "scaled_down_2d_mono", False, False)
    save(up_img, "scaled_up_2d_mono", False, False)


def check_2D_color():
    IMAGE_PATH = "/mnt/asgard2/code/nadim/case2d_code_copy/support.scripts/5af89c8c.jpg"
    img = np.array(Image.open(IMAGE_PATH))
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = img[tf.newaxis, :, :, :]

    br = ResizingOperator(tf.shape(img), method=ResizingMethods.NEAREST_NEIGHBOR)
    down_img = br.resize_down(img, [2, 2])
    up_img = br.resize_down(img, [2, 2])

    save(img, "original_2d_color", True, False)
    save(down_img, "scaled_down_2d_color", True, False)
    save(up_img, "scaled_up_2d_color", True, False)


check_2D_mono()
check_2D_color()
check_3D()
