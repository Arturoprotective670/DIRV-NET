"""
This script aims to interpolate 4DCT data to a unified resolution.
"""

# below is needed to help python references lookups when this
# script is run separately, source https://stackoverflow.com/a/4383597
import sys

STARTUP_DIR = (
    "/mnt/asgard2/code/nadim/case2d_code_copy/"  # change this absolute path to yours
)
sys.path.insert(1, STARTUP_DIR)

from support.scripts.file_formats.h5_format import *
from source.framework.tools.pather import Pather
from source.model.tools.operators.resizing_operator import (
    ResizingOperator,
    ResizingMethods,
)
from source.model.tools.shape_break_down import ShapeBreakDown
import tensorflow as tf

DIR_h5 = "/mnt/asgard2/data/nadim/h5_my_uncompressed/h5"
OUTPUT_DIR = "/mnt/asgard2/data/nadim/h5_my_interpolated_2x2x2/"
DTYPE = tf.float32

# cases resolutions, as per
# https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html
voxel_dimensions_mm = []
voxel_dimensions_mm.append([0.97, 0.97, 2.5])  # X,Y,Z
voxel_dimensions_mm.append([1.16, 1.16, 2.5])
voxel_dimensions_mm.append([1.15, 1.15, 2.5])
voxel_dimensions_mm.append([1.13, 1.13, 2.5])
voxel_dimensions_mm.append([1.10, 1.10, 2.5])
voxel_dimensions_mm.append([0.97, 0.97, 2.5])
voxel_dimensions_mm.append([0.97, 0.97, 2.5])
voxel_dimensions_mm.append([0.97, 0.97, 2.5])
voxel_dimensions_mm.append([0.97, 0.97, 2.5])
voxel_dimensions_mm.append([0.97, 0.97, 2.5])

target_pixel_resolution_mm = [2, 2, 2]
target_pixel_resolution_mm = tf.convert_to_tensor(
    target_pixel_resolution_mm, dtype=DTYPE
)

output_dir_pather = Pather(OUTPUT_DIR)

for file in Pather(DIR_h5).contents("*.h5"):
    data = read_h5(file)
    data = tf.convert_to_tensor(data, dtype=DTYPE)
    data = data[tf.newaxis, :, :, :, tf.newaxis]  # to convert to batch format

    file_pather = Pather(file)
    case_id, phase_id = file_pather.name.split("_")
    dimensions = voxel_dimensions_mm[int(case_id) - 1]
    # dimensions.reverse() # make Z,Y,X
    voxel_dimensions = tf.convert_to_tensor(dimensions, dtype=DTYPE)

    shape = ShapeBreakDown(data)

    scale_factors = target_pixel_resolution_mm / voxel_dimensions
    scale_factors = list(scale_factors.numpy())

    resizer = ResizingOperator(shape.batch_shape, method=ResizingMethods.B_SPLINES)
    data = resizer.resize_down(data, scale_factors)
    data = data[0, ..., 0].numpy()

    new_file_path = output_dir_pather.join(file_pather.full_name)
    write_h5(new_file_path, data)
    print(
        f"The file {new_file_path} is interpolated from {shape.core_shape_list} to {data.shape}."
    )
