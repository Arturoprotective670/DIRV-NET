# below is needed to help python references lookups when this
# script is run separately, source https://stackoverflow.com/a/4383597
import sys

STARTUP_DIR = (
    "/mnt/asgard2/code/nadim/case2d_code_copy/"  # change this absolute path to yours
)
sys.path.insert(1, STARTUP_DIR)


from support.scripts.file_formats.nrrd_format import read_nrrd

SOURCE_FILE = "data/nrrd_experiment/source/1715832_F.nrrd"
data = read_nrrd(SOURCE_FILE)
print("source data size: " + str(data.shape))

from support.scripts.nrrd_helper.pre_process_nrrd_image import pre_process_image
from tools.slices_generator import *

SLICES_TARGET_DIR = "data/nrrd_experiment/slices"

# below is for debug purposes, i.e. testing slicing
# randomly_slice_array_to_images(data,
#                 slices_target_dir,
#                 slicing_axis = 0,
#                 slices_count = 10,
#                 pre_process_image = pre_process_image)

slices = array_random_slicer(
    data, slicing_axis=0, slices_count=240, pre_process_image=pre_process_image
)

def random_array_patches_to_images(
    data: array,
    target_dir: str,
    patch_size: Tuple,
    patches_count: int,
    auto_adapt_patches_count: bool = False,
    random_seed: int = 42,
    file_name_prefix: str = "patch_",
    initial_file_index: int = 0,
) -> None:
    """
    Slices a 3D monochrome image array to random 2D samples/slices and save them
    to image jpg files.

    Arguments
    ---------
    - `data`: the 3D array of monochrome image.
    - `target_dir`: the directory in whish the 2D jpg slices will be saved. If the
    directory dose not exists, it will be created.
    - `patches_count`: the number of patches to create, should be smaller than any
    of array dimensions.
    - `auto_adapt_patches_count`: patches_count will be reduced if the requested
    count is bigger than data array size.
    - `pre_process_image`: an optional function, to pre-process the image before
    saving it, takes and array as an argument, and returns an array.
    - `file_name_prefix`: a string that file names will start with.
    - `initial_file_index`: the initial index that will be appended to images file
    name, usefully if the slicing results along different axis will be saved in
    the same directory.
    - `random_seed`: a seed that can fixes for reproducibility.

    Return
    ------
    - None
    """

    patches = random_array_patches(
        data, patch_size, patches_count, auto_adapt_patches_count, random_seed
    )

    from os import makedirs
    from os.path import join, exists

    if not exists(target_dir):
        makedirs(target_dir)

    index = initial_file_index

    from PIL import Image

    for patch in patches:
        image = Image.fromarray(patch)
        image = image.convert("L")  # source: https://stackoverflow.com/a/51909646

        file_name = join(target_dir, file_name_prefix + str(index) + ".jpg")
        image.save(file_name)

        index += 1


from tools.patches_generator import *

PATCHES_TARGET_DIR = "data/nrrd_experiment/patches"
slice_index = 0
for slice in slices:
    # avoid zero length slices
    if len(slice) > 0:
        random_array_patches_to_images(
            slice,
            PATCHES_TARGET_DIR,
            patch_size=(84, 84),
            patches_count=25,
            auto_adapt_patches_count=True,
            file_name_prefix="slice_{}_patch_".format(slice_index),
        )
    slice_index += 1
