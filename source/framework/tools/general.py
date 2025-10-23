"""
Purpose
--------
- Check if python script running in debug mode.
- Calc the batch per item mean norm.
- Calc the block average of sequence of numbers.
- Calc the area under discrete (line-wise connected) curve.
- Calc and combine array/list/Tensor statists.
- Print of any object properties and their values.

Contributors
------------
- TMS-Namespace
"""

import copy
import math
from typing import List, Optional, Union
from numpy.typing import NDArray

import numpy as np

import tensorflow as tf

from source.model.tools.shape_break_down import ShapeBreakDown


def is_debug_mode() -> bool:
    """
    Check if the code running in the debug mode.

    Returns
    --------
    - True if debug mode is detected, otherwise returns False.

    Notes
    ------
    - This theoretically should work with various IDE's such as VS Code and PyCharm.

    Source
    ------
    - https://stackoverflow.com/a/67065084
    """
    import sys

    return hasattr(sys, "gettrace") and sys.gettrace() is not None


def get_dissimilarity(tensor1: tf.Tensor, tensor2: tf.Tensor) -> tf.Tensor:
    """
    Returns for each batch item, the mean of last dimension norm... (aka. dissimilarity per image/item)
    """
    # we calc the diff, then we square it, and calculate the mean of last
    # dimension, which is the flow field components, or image channels, then we
    # take the root, to get the per pixel norm mean
    pixel_norm_mean = tf.sqrt(tf.reduce_mean(tf.square(tensor1 - tensor2), axis=-1))

    # finally, we calc the mean per image/batch item, for that we need to
    # exclude the batch dimension
    rank = tf.rank(tensor1)
    if rank == 4:
        axis = [1, 2]
    elif rank == 5:
        axis = [1, 2, 3]
    else:
        raise NotImplementedError

    return tf.reduce_mean(pixel_norm_mean, axis=axis)


def block_mean(
    data: List[float], block_size: int, force_all_items: bool = True
) -> List:
    """
    Calculates the mean of a list, for every specified number of elements.

    Arguments
    ---------
    - `data`: the list on which the block mean is calculated.
    - `block_size`: specifies the number of elements that the average will be
    calculated on.
    - `force_all_items`: if `true`, and the number of elements in the provided
    data, is not enough to divide it to an integer number of blocks, an average
    of the remaining elements will be added to the result.

    Returns
    -------
    - a list of the calculated block means.
    """

    if block_size == 1:
        return data

    size = len(data)

    if block_size > size:
        block_size = size

    limit_to = (size // block_size) * block_size

    limited_data = np.array(data[:limit_to])
    limited_data = limited_data.reshape(-1, block_size)

    result = np.mean(limited_data, axis=1)

    if isinstance(result, float):
        result = [result]
    else:
        result = list(result)

    if force_all_items and limit_to != size:
        limited_data = np.array(data[:limit_to])
        result += [np.mean(limited_data)]

    return result


def discrete_curve_integral(
    y: List[float], x_from: float = 0, x_to: float = 1
) -> List[float]:
    knots_count = len(y)

    x = np.linspace(x_from, x_to, knots_count)

    accumulative_area = [0.0]

    dy = np.diff(y)
    dx = np.diff(x)

    for i in range(knots_count - 1):
        if y[i] * y[i + 1] >= 0:
            # trapezoid  area
            area = (y[i] + y[i + 1]) * dx[i] / 2
        else:  # we passing by a root
            root_x = x[i] - y[i] * dx[i] / dy[i]
            # two triangles area
            area = ((root_x - x[i]) * y[i] + (x[i + 1] - root_x) * y[i + 1]) / 2

        accumulative_area.append(area + accumulative_area[i])

    return accumulative_area


def gaussian_kernel(
    rank: int, size: int, sigma: float, datatype: tf.DType = tf.float32
) -> tf.Tensor:
    """
    Generates a discrete Gaussian kernel.

    Parameters
    ----------
    - `rank`: An integer, 2 for 2D kernel, 3 for 3D kernel.
    - `size`: The size of the kernel.
    - `sigma`: A float, the standard deviation of the Gaussian distribution.
    - `datatype`: The data type of the tensor elements in the kernel.

    Returns
    -------
    - The Gaussian kernel tensor.
    """
    size = int(size) // 2

    rng = tf.range(-size, size + 1)
    axis_grids = tf.meshgrid(*([rng] * rank))

    for axis_index in range(rank):
        axis_grids[axis_index] = tf.cast(tf.square(axis_grids[axis_index]), datatype)

    norms = tf.stack(axis_grids, axis=-1)
    norms = tf.reduce_sum(norms, axis=-1)

    un_normalized = tf.exp(-norms / (2 * sigma**2))

    return tf.constant(un_normalized / tf.reduce_sum(un_normalized))


def truncated_white_noise(
    batch_shape: ShapeBreakDown,
    random_generator: np.random.Generator,
    max_value: float,
    float_data_type: tf.DType = tf.float32,
) -> tf.Tensor:
    """
    Generates white noise, which has a maximum value.
    """
    shape = (
        [batch_shape.batch_size_int]
        + batch_shape.core_shape_list
        + [batch_shape.core_rank_int]
    )

    noise = random_generator.normal(0, 1, shape)
    mx = np.max(np.abs(noise))
    noise = (noise / mx) * max_value

    return tf.convert_to_tensor(noise, dtype=float_data_type)


def random_binary_choice_indexes(
    count: int,
    false_probability: float,
    random_generator: np.random.Generator,
) -> List[int]:
    """
    Return a list of indexes, where selecting each index has 1 - false_probability chance.
    """
    choice_prob = (1 - false_probability, false_probability)
    random_bools = random_generator.choice([True, False], size=count, p=choice_prob)

    false_indexes = [i for i, val in enumerate(random_bools) if not val]

    return false_indexes


class StatisticsData:
    def __init__(
        self,
        size: int,
        mx: float,
        mn: float,
        mean: float,
        std: float,
        std_percent: float,
        dissimilarity: float,
    ) -> None:
        self.maximum = mx
        self.minimum = mn
        self.mean = mean
        self.standard_deviation = std
        self.coefficient_of_variation = std_percent
        self.size = size
        self.dissimilarity = dissimilarity

    def clone(self) -> "StatisticsData":
        cp = copy.copy(self)
        return cp


def calc_statistics(
    data: Union[NDArray, List, tf.Tensor],
) -> StatisticsData:
    """
    Calcs different *biased* statistics of arrays, lists, or tensors.

    Notes
    -----
    - standard deviation in numpy and tensorflow are biased by default.

    Parameters
    ----------
    - `data`: An array, a list, or a tensor.

    Returns
    -------
    - `StatisticsData` object.
    """
    if isinstance(data, tf.Tensor):
        mx = tf.reduce_max(data).numpy()
        mn = tf.reduce_min(data).numpy()

        mean = tf.reduce_mean(data).numpy()

        std = tf.math.reduce_std(data).numpy()
        std_percent = 100 * (std / mean)

        size = tf.size(data).numpy()

        data = _make_suitable_for_dissimilarity(data)
        dissimilarity = get_dissimilarity(data, tf.zeros_like(data))
    else:
        if isinstance(data, list):
            data = np.array(data)

        mx = np.max(data)
        mn = np.min(data)

        mean = np.mean(data)

        std = np.std(data)
        std_percent = 100 * (std / mean)

        size = np.size(data)

        data_tensor = tf.convert_to_tensor(data)
        data_tensor = _make_suitable_for_dissimilarity(data_tensor)

        dissimilarity = get_dissimilarity(data_tensor, tf.zeros_like(data_tensor))

    dissimilarity = tf.reduce_mean(dissimilarity).numpy()

    return StatisticsData(size, mx, mn, mean, std, std_percent, dissimilarity)


def _make_suitable_for_dissimilarity(tensor: tf.Tensor) -> tf.Tensor:
    if tf.rank(tensor) == 3:
        tensor = tf.expand_dims(tensor, axis=0)
    elif tf.rank(tensor) == 2:
        tensor = tf.expand_dims(tensor, axis=0)
        tensor = tf.expand_dims(tensor, axis=-1)
    elif tf.rank(tensor) == 1:
        tensor = tf.expand_dims(tensor, axis=0)
        tensor = tf.expand_dims(tensor, axis=-1)
        tensor = tf.expand_dims(tensor, axis=-1)

    return tensor


def combine_statistics(statistics_list: List[StatisticsData]) -> StatisticsData:
    """
    Combines statistical estimators from a list of `StatisticsData` objects.

    Notes
    -----
    - This function assumes biased statists.

    References
    ----------
    - combining standard deviations: https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups

    Returns
    -------
    - `StatisticsData` object.
    """
    combined_maximum = max(stat.maximum for stat in statistics_list)
    combined_minimum = min(stat.minimum for stat in statistics_list)

    combined_size = sum(stat.size for stat in statistics_list)

    # combined mean
    combined_mean = (
        sum(stat.mean * stat.size for stat in statistics_list) / combined_size
    )

    # set to 1 if you want unbiased std
    ddof = 0

    # combined variance
    combined_variance_numerator = sum(
        (stat.size - ddof) * stat.standard_deviation**2
        + stat.size * (stat.mean - combined_mean) ** 2
        for stat in statistics_list
    )

    combined_variance = combined_variance_numerator / (
        combined_size - len(statistics_list) * ddof
    )

    # combined standard deviation
    combined_std = math.sqrt(combined_variance)

    dissimilarity = (
        sum(stat.dissimilarity * stat.size for stat in statistics_list) / combined_size
    )

    return StatisticsData(
        combined_size,
        combined_maximum,
        combined_minimum,
        combined_mean,
        combined_std,
        combined_std / combined_mean * 100,
        dissimilarity,
    )


def print_object(obj: object) -> None:
    properties = [
        (attribute, value)
        for attribute, value in sorted(vars(obj).items())
        if not attribute.startswith("_")
    ]

    max_attr_length = max(len(attr) for attr, _ in properties)
    max_val_length = max(len(str(val)) for _, val in properties)

    print(f"{'Property'.ljust(max_attr_length)}  {'Value'.ljust(max_val_length)}")
    print(f"{'-' * max_attr_length}  {'-' * max_val_length}")

    for attribute, value in properties:
        print(f"{attribute.ljust(max_attr_length)}  {str(value).ljust(max_val_length)}")
