from_dir = "/mnt/asgard2/code/nadim/case2d_code_copy/data/nrrd_experiment/patches"
to_dir = "/mnt/asgard2/code/nadim/case2d_code_copy/data/nrrd_experiment/patches28"


import glob

files = glob.glob(from_dir + "/*.jpg")

from PIL import Image

image_dim = 28

from pathlib import Path
from os.path import join, exists
from os import makedirs

if not exists(to_dir):
    makedirs(to_dir)

for f in files:
    image = Image.open(f)
    image = image.resize((image_dim, image_dim), Image.ANTIALIAS)
    image.save(join(to_dir, Path(f).stem) + ".jpg")
