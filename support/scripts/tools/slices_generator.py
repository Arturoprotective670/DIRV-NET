from typing import List, Callable
from array import array

import random


def slice_array(data: array, slicing_axis: int, slice_index: int) -> array:
    """
    Gets a N-1 dimensional slice of a N-dimensional array.

    Arguments
    ---------
    - `data`: the array to slice.
    - `slicing_axis`: the index along which slicing will be performed.
    - `slice_index`: the index of the slice to return, along slicing_axis.

    Returns
    --------
    - An array that represents the requested slice.
    """
    # source https://stackoverflow.com/a/15488285
    slicing_range = [slice(None)] * data.ndim

    if slicing_axis >= data.ndim - 1:
        raise Exception("Slicing axis is out of array dimensionality.")

    slicing_range[slicing_axis] = slice_index
    return data[tuple(slicing_range)]


def array_random_slicer(
    data: array,
    slicing_axis: int,
    slices_count: int,
    pre_process_image: Callable[[array], array] = None,
) -> List[array]:
    """
    Generates N-1 dimensional random slices, from a N-dimensional array.

    Arguments
    ---------
    - `data`: the array to slice.
    - `slicing_axis`: the index along which slicing will be performed.
    - `slices_count`: the number of random slices to generate.
    - `pre_process_image`: an optional function, to pre-process the image.

    Returns
    -------
    - A list of arrays, that represents a list of array slices.
    """

    if data.shape[slicing_axis] < slices_count:
        raise Exception("Requested slices count larger than source data size.")

    random_slicing_indexes = random.sample(
        range(data.shape[slicing_axis]), slices_count
    )

    if pre_process_image is not None:
        return [
            pre_process_image(slice_array(data, slicing_axis, index))
            for index in random_slicing_indexes
        ]
    else:
        return [
            slice_array(data, slicing_axis, index) for index in random_slicing_indexes
        ]


def randomly_slice_array_to_images(
    data: array,
    target_dir: str,
    slicing_axis: int,
    slices_count: int,
    pre_process_image: Callable[[array], array] = None,
    file_name_prefix: str = "slice_",
    initial_file_index: int = 0,
) -> None:
    """
    Slices a 3D monochrome image array to random 2D samples/slices and save them to image files.

    Arguments
    ---------
    - `data`: the 3D array of monochrome image.
    - `target_dir`: the directory in whish the 2D jpg slices will be saved. If the
    directory dose not exists, it will be created.
    - `slices_count`: the number of slices to create, should be smaller that the
    corresponding image axis size.
    - `pre_process_image`: an optional function, to pre-process the image before
    saving it, takes and array as an argument, and returns an array.
    - `file_name_prefix`: a string that file names will start with.
    - `initial_file_index`: the initial index that will be appended to images file name,
    usefully if the slicing results along different axis will be saved in the same directory.

    Return
    ------
    - None
    """

    if data.ndim < 3:
        raise Exception("Data should be a 3D image array.")

    slices = array_random_slicer(data, slicing_axis, slices_count, pre_process_image)

    print("slicing finished, starting image saving...")

    from os import makedirs
    from os.path import join, exists

    if not exists(target_dir):
        makedirs(target_dir)

    index = initial_file_index

    from PIL import Image

    for slice in slices:
        # sometimes pre_process_image may return empty array due to cropping
        if len(slice) > 0:
            image = Image.fromarray(slice)
            image = image.convert("L")  # source: https://stackoverflow.com/a/51909646

            file_name = join(target_dir, file_name_prefix + str(index) + ".jpg")
            image.save(file_name)
        else:
            print("WARNING: an empty image is detected!")

        index += 1

    print("all images are saved.")
