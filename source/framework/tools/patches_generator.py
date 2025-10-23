"""
Purpose
-------
- Generates random patches from a 2D/3D array.

Contributors
------------
- TMS-Namespace
"""

from typing import List
import numpy as np
from numpy.typing import NDArray


def random_array_patches(
    data: NDArray,
    patch_shape: List,
    patches_count: int,
    auto_adapt_patches_count: bool = False,
    random_generator : np.random.Generator = np.random.default_rng()
) -> List[NDArray]:
    """
    Creates random patches from a 2D/3D array.

    Arguments
    ---------
    - `data`: the `array` data.
    - `patch_size`: the desired size of patches.
    - `patches_count`: the number of patches to generate.
    - `auto_adapt_patches_count`: indicates if the `patches_count` can be auto
    reduced if the requested count is bigger than data array size.
    - `random_generator`: the numpy random generator to use.

    Returns
    -------
    - A `list` of the patches `array`s.

    Notes
    -----
    - The array slicing dimensions for patch generation, are decided by
    `patch_shape` rank, so the `data` can be an image with channels dimension.
    """

    data_shape = data.shape
    images_rank = len(patch_shape)

    for dim in range(images_rank):
        if patch_shape[dim] >= data_shape[dim]:
            raise Exception("Patch size is too big.")

    # to avoid patches going out of image boundaries, we reduce the range of
    # possible patch corner location
    coordinate_ranges = []
    coordinate_range_stops = []  # ranges stops, to make things easier
    is_patches_count_bigger = True  # we need to know if it is bigger than all ranges

    for dim in range(images_rank):
        dim_range = range(data_shape[dim] - patch_shape[dim])
        coordinate_ranges.append(dim_range)
        coordinate_range_stops.append(dim_range.stop)
        is_patches_count_bigger = is_patches_count_bigger and (
            patches_count > dim_range.stop
        )

    if is_patches_count_bigger:
        if auto_adapt_patches_count:
            # patches count should be at most equal to biggest range, this will be
            # enough to keep patches location random at least at one dimension
            patches_count = max(coordinate_range_stops)
        else:
            raise Exception("Requested patches count is too big for this array size.")

    patches_coordinates = []
    for dim in range(images_rank):
        # if is_patches_count_bigger is True,  patches_count may be bigger than
        # range end, then we need to enable replacing/repeating in choices
        patches_coordinates.append(
            random_generator.choice(
                coordinate_ranges[dim],
                patches_count,
                replace=patches_count > coordinate_range_stops[dim],
            )
        )

    patches = []

    y_coordinates = patches_coordinates[0]
    x_coordinates = patches_coordinates[1]

    if images_rank == 2:
        for index in range(patches_count):
            patch = data[
                y_coordinates[index] : (y_coordinates[index] + patch_shape[0]),
                x_coordinates[index] : (x_coordinates[index] + patch_shape[1]),
                ...,
            ]
            patches.append(np.copy(patch))

    elif images_rank == 3:
        z_coordinates = patches_coordinates[2]

        for index in range(patches_count):
            patch = data[
                y_coordinates[index] : (y_coordinates[index] + patch_shape[0]),
                x_coordinates[index] : (x_coordinates[index] + patch_shape[1]),
                z_coordinates[index] : (z_coordinates[index] + patch_shape[2]),
                ...,
            ]
            patches.append(np.copy(patch))
    else:
        raise NotImplementedError

    return patches
