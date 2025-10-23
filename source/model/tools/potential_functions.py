"""
Purpose
-------
- Interpolate a line-wise discrete function values at sampling points.

Contributors
------------
- TMS-Namespace
- Claudio Fanconi
"""

import tensorflow as tf


def potential_function_piece_wise_linear(
    sampling_points: tf.Tensor,
    knots: tf.Tensor,
    min_x: float = 0.0,
    max_x: float = 1.0,
    int_type: tf.DType = tf.int32,
    cropping_offset: float = 1e-5,
) -> tf.Tensor:
    """
    Calculates the interpolated values of the piecewise linearly parametrized
    functions on the sampling points, on a per-kernel manner.

    Arguments
    ---------
    - `sampling_points`: Tensor of `[B, core_size, filters_count]` size,
      represents the points that we want to get the interpolated values at.
    - `knots`: Tensor of size `[knots_count, filters_count]`, represents the
      values of piecewise knots of the functions, per filter.

    Returns
    --------
    - The linearly interpolated functions
    """

    knots_count = tf.shape(knots)[0]
    filters_count = tf.shape(knots)[1]
    initial_sampling_points_shape = tf.shape(sampling_points)

    sampling_points = tf.reshape(sampling_points, [-1, filters_count])
    samples_per_filter = tf.shape(sampling_points)[0]

    knots_count = tf.cast(knots_count, dtype=sampling_points.dtype)

    interpolation_grid = tf.range(filters_count)
    interpolation_grid = tf.cast(interpolation_grid, dtype=sampling_points.dtype)
    interpolation_grid = tf.reshape(interpolation_grid, [1, filters_count])
    interpolation_grid = tf.tile(interpolation_grid, [samples_per_filter, 1])

    weight = (max_x - min_x) / (knots_count - 1.0)

    data_normalized = (sampling_points - min_x) / weight
    data_normalized = tf.clip_by_value(
        data_normalized, cropping_offset, knots_count - (1 + cropping_offset)
    )

    data_normalized_floor = tf.floor(data_normalized)
    delta = data_normalized - data_normalized_floor

    idx_floor = data_normalized_floor
    idx_ceiling = data_normalized_floor + 1

    nd_floor = tf.stack([idx_floor, interpolation_grid], axis=2)
    nd_ceiling = tf.stack([idx_ceiling, interpolation_grid], axis=2)

    nd_floor = tf.cast(nd_floor, dtype=int_type)
    nd_ceiling = tf.cast(nd_ceiling, dtype=int_type)

    y_floor = tf.gather_nd(knots, nd_floor)
    y_ceiling = tf.gather_nd(knots, nd_ceiling)

    y = y_floor * (1 - delta) + delta * y_ceiling
    y = tf.reshape(y, initial_sampling_points_shape)

    return y


def potential_function_RBF(
    data,
    weights,
    filters_count: int,
    int_type=tf.int32,
    min_x: float = 0.0,
    max_x: float = 1.0,
):
    """Use RBF to approximate the potential function
    Args:
        xin: Input
        yK: learnable activation functions
        interp_grid: interpolation grid
        n_flt: number of filers
        minx (default 0.0): min input value
        maxx (default 1.0): max input value
    Returns:
        RBF interpolated output
    """

    # TODO: upgrade to 3D
    interpolation_knots_count = weights.shape[0]

    interpolation_grid = tf.linspace(0.0, 1.0, interpolation_knots_count)
    interpolation_grid = tf.reshape(
        interpolation_grid, [1, 1, interpolation_knots_count]
    )

    sigma = (max_x - min_x) / (interpolation_knots_count - 1)

    original_shape = tf.shape(data)
    data = tf.reshape(data, [-1, filters_count, 1])

    pw_dst = tf.square((data - interpolation_grid))
    exponents = tf.exp(-pw_dst / (2 * sigma**2))
    weights = tf.reshape(weights, [1, filters_count, interpolation_knots_count])

    interpolated = tf.reduce_sum(exponents * weights, axis=2, keepdims=False)
    interpolated = tf.reshape(interpolated, original_shape)

    return interpolated
