"""
Purpose
-------
- Calculating mapping coordinates between two tensors.
- Perform B-Spline interpolation.
- Used to work with control points mainly in this project.

Reference
---------
- This is a 2D/3D version of `tfg.math.interpolation.bspline.interpolate`, and is
adapted from:
https://github.com/sarthakksu/covid-low-income-bam/blob/a9ab0de05b525db3137570989fc4cef64130d659/factorize_a_city/libs/image_alignment.py

Contributors
------------
- TMS-Namespace
- Claudio Fanconi
"""

from typing import List, Tuple

import tensorflow as tf
from tensorflow_graphics.math.interpolation import bspline

from source.model.tools.shape_break_down import ShapeBreakDown


def batch_b_splines_interpolation(
    knots: tf.Tensor,
    positions: tf.Tensor,
    degree: int,
    cyclical: Tuple[bool, bool],
    int_data_type: tf.DType,
) -> tf.Tensor:
    """
    Interpolates the `knot` values at `positions` of a B-spline surface warp.

    Arguments
    ---------
    - `knots`: A tensor with shape [bsz, KH, KW, KCh] representing the values to
      be interpolated over. In warping these are control_points.
    - `positions`: A tensor with shape [bsz, H, W, 2] that defines the desired
        positions to interpolate. Positions must be between [0, KHW - D) for
        non-cyclical and [0, KHW) for cyclical splines, where KHW is the number
        of knots on the height or width dimension and D is the spline degree.
        The last dimension of positions record [y, x] coordinates.
    - `degree`: An int describing the degree of the spline.
        There must be at least D+1 horizontal and vertical knots.
    - `cyclical`: A length-two tuple bool describing whether the spline is
      cyclical in the height and width dimension respectively.

    Returns:
    --------
    - A tensor of shape '[bsz, H, W, KCh]' with the interpolated value based on
      the control points at various positions.

    Raises:
    -------
    - `ValueError`: If degree is greater than 4 or num_knots - 1, or less than 0.
    - `InvalidArgumentError`: If positions are not in the right range.

    Notes
    -----
    - This is a sparse mode B-spline warp, so the memory usage is efficient.
    """
    knots_shape = ShapeBreakDown(knots)

    weights = []
    indexes = []

    for dim in tf.range(knots_shape.core_rank):
        control_points_weights, knots_index = bspline.knot_weights(
            positions=positions[..., dim],
            num_knots=int(knots_shape.core_shape[dim]),
            degree=int(degree),
            cyclical=cyclical[dim],
            sparse_mode=True,
        )

        weights.append(control_points_weights)

        if cyclical[dim]:
            stacked_indexes = []

            for i in tf.range(-degree // 2, degree // 2 + 1):
                stacked_indexes.append(knots_index + i)

            stacked_indexes = tf.stack(stacked_indexes, axis=-1)
            stacked_indexes = tf.math.floormod(
                stacked_indexes, knots_shape.core_shape[dim]
            )
        else:
            stacked_indexes = [knots_index + i for i in range(degree + 1)]
            stacked_indexes = tf.stack(stacked_indexes, axis=-1)

        indexes.append(stacked_indexes)

    indexes = _unify_shape(_extend_rank(indexes))

    # combine the weights together, by simply multiplying them (after extending
    # their rank)
    weights = _extend_rank(weights)
    mixed = 1
    for i in range(knots_shape.core_rank_int):
        mixed *= weights[i]

    # create batch indexes
    batch_ind = tf.range(
        0, knots_shape.batch_size, 1
    )  # range with dtype=int16 raises error
    batch_ind = tf.cast(batch_ind, dtype=int_data_type)
    for i in range(knots_shape.core_rank_int * 2):  # make the rank as in indexes
        batch_ind = tf.expand_dims(batch_ind, axis=-1)
    # grow the size as in indexes
    batch_ind += tf.zeros_like(indexes[0])

    # tf.gather process dimensions left to right which means (batch, H, W)
    gather_idx = tf.stack(indexes, axis=-1)
    idx_new_shape = [knots_shape.batch_size, -1, knots_shape.core_rank]
    gather_nd_indices = tf.reshape(gather_idx, idx_new_shape)

    relevant_cp = tf.gather_nd(knots, gather_nd_indices, batch_dims=1)
    cp_new_shape = gather_idx.shape.as_list()[:-1] + [knots_shape.channels_count_int]
    reshaped_cp = tf.reshape(relevant_cp, cp_new_shape)

    mixed = mixed[..., tf.newaxis]

    return tf.reduce_sum(
        reshaped_cp * mixed, axis=[-i for i in range(2, knots_shape.batch_rank_int)]
    )


def batch_coordinates_map(
    source_size: tf.Tensor,
    target_size: tf.Tensor,
    batch_size: tf.Tensor,
    float_data_type: tf.DType,
) -> tf.Tensor:
    """
    Creates a map of coordinates for a batch, by using which, one can map target
    coordinates to the source coordinates.

    Arguments
    ---------
    - `source_size`: a `Tensor` of the grid size of the source that represents
    our base for mapping.
    - `target_size`: a `Tensor` the size of the target that we want to map from
    to the source tensor.
    - `batch_size`: the size of the needed batch.

    Returns
    -------
    - a tensor of `[target_size, images_rank]` that contains the corespondent
    coordinates in the source coordinates. Can be used for mapping from the
    target to the given source coordinates.
    """
    # TODO: add support for channels
    start = tf.constant(0.0, dtype=float_data_type)
    stop = tf.cast(source_size, dtype=float_data_type) - 1 - 1e-6
    rank = len(source_size)

    coordinates = []

    for dim in tf.range(rank):
        coordinates.append(tf.linspace(start, stop[dim], target_size[dim]))

    coordinates = _unify_shape(_extend_rank(coordinates))

    stacked_coords = tf.stack(coordinates, axis=-1)[tf.newaxis]
    tiling_multiples = tf.constant([int(batch_size)] + ([1] * (rank + 1)))

    return tf.tile(stacked_coords, tiling_multiples)


def _unify_shape(tensors: List[tf.Tensor]) -> List[tf.Tensor]:
    """
    Unifies the sizes of each dimension of a list of tensors, by inserting
    zeros.

    Notes
    -----
    - It assumes that all tensors are of the same rank.
    """
    dtype = tensors[0].dtype
    range = tf.range(len(tensors))
    # we unify dimensions by adding zero-tensors of the shape of other list tensors
    for i in range:
        for j in range:
            tensors[i] += tf.zeros_like(tensors[j], dtype=dtype)

    return tensors


def _extend_rank(tensors: List[tf.Tensor]) -> List[tf.Tensor]:
    """
    Adds a complimentary dimensions to the end of tensor, such that all tensors
    in the list can hold similar information.

    Returns
    -------
    - A list of tensors, of [..., Y,X] or [..., Z,Y,X] structure (depending on
    the number of tensors in the list).

    Notes
    -----
    - It assumes that the tensors in the list are ordered in correspondence of
    the output (i.e. first one is [..., Z], second one [..., Y] etc..).
    - The added dimensions are complimentary (ex. for first tensor in the list
    of size three, the added dimensions are for X and Z, so it will be [...,
    Z,Y,X]).
    """
    rank = len(tensors)

    if rank == 2:
        extend = [[-1], [-2]]
    elif rank == 3:
        extend = [[-1, -1], [-2, -1], [-2, -2]]
    else:
        raise NotImplementedError

    for i in range(rank):
        for j in range(len(extend[0])):
            tensors[i] = tf.expand_dims(tensors[i], axis=extend[i][j])

    return tensors
