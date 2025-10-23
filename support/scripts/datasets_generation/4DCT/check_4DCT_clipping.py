# below is needed to help python references lookups when this
# script is run separately, source https://stackoverflow.com/a/4383597
import sys

STARTUP_DIR = (
    "/mnt/asgard2/code/nadim/case2d_code_copy/"  # change this absolute path to yours
)
sys.path.insert(1, STARTUP_DIR)


FILE_PATH = (
    "/mnt/asgard2/code/nadim/case2d_code_copy/data/h5_experiment/origin/4DCT_case1.h5"
)
# according to Isotropic Total Variation Regularization of
# Displacements in Parametric Image Registration
# Valeriy Vishnevskiy, Christine Tanner, Orcun Goksel et. al. paper
MAX_CLIP = 1200
MIN_CLIP = 50
from support.scripts.file_formats.h5_format import read_h5
import numpy as np
from PIL import Image

res = read_h5(FILE_PATH)
first_pase = res[..., 0]
slice_pos = first_pase.shape[-1] // 2
img_2d = first_pase[..., slice_pos]

img = img_2d / np.max(res)
img = Image.fromarray(np.uint8(img * 255)).convert("L")
img.save("slice_not_clipped.jpg")

img = np.clip(img_2d, MIN_CLIP, MAX_CLIP)
img = (img - MIN_CLIP) / (MAX_CLIP - MIN_CLIP)
img = Image.fromarray(np.uint8(img * 255)).convert("L")
img.save("slice_clipped.jpg")


# for i in range(res.shape[-1]):
#     r = res[... , i]#[96:160, 96:160, 16: 80]
#     r = np.clip(r, MIN_CLIP, MAX_CLIP)
#     r =  (r - MIN_CLIP) / (MAX_CLIP - MIN_CLIP)
