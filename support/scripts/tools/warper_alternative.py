"""
Notes
-----
- This is an alternative implementation of the warper, before I found that some
  TF libraries has it's own implementation.

Contributors
------------
- TMS-Namespace
- Claudio Fanconi
"""

import tensorflow as tf

from source.model.tools.mesh_generator import batch_mesh_grid
from source.model.tools.shape_break_down import ShapeBreakDown


def inverse_warp(images: tf.Tensor, flows: tf.Tensor) -> tf.Tensor:
    shape = ShapeBreakDown(images)

    original_grid = batch_mesh_grid(shape.core_shape, shape.batch_size_int)
    sampling_grid = original_grid + flows

    if shape.core_rank_int == 3:
        return _trilinear_interpolation(images, sampling_grid, shape)
    elif shape.core_rank_int == 2:
        return _bilinear_interpolation(images, sampling_grid, shape)
    else:
        raise NotImplementedError


def _bilinear_interpolation(
    images: tf.Tensor,
    sampling_grid: tf.Tensor,
    shape: ShapeBreakDown,
    int_data_type=tf.int32,
) -> tf.Tensor:
    """
    Direct trilinear interpolation, from sampling points.

    Arguments
    ---------
    - `images`: batch of 3D images to interpolate.
    - `sampling_grid`:
    Returns:
        image transformed according to the displacement vector field
    """
    max_y = shape.height_int - 1
    max_x = shape.width_int - 1

    # if tf.math.reduce_all(p == 0):
    #     return img

    x = sampling_grid[:, :, :, 0]
    y = sampling_grid[:, :, :, 1]

    x0 = tf.floor(x)
    x1 = x0 + 1
    y0 = tf.floor(y)
    y1 = y0 + 1

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # note that casting should be done after weights/deltas calculation, to
    # avoid zeros in deltas due to rounding, since they will cause artifacts
    x0 = tf.cast(x0, int_data_type)
    x1 = tf.cast(x1, int_data_type)
    y0 = tf.cast(y0, int_data_type)
    y1 = tf.cast(y1, int_data_type)

    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)

    # === generate batches only indexes, to be blended with other indexes
    batches_indexes = tf.range(0, shape.batch_size)
    batches_indexes = tf.reshape(
        batches_indexes, [shape.batch_size_int] + [1] * shape.core_rank_int
    )
    batches_indexes = tf.tile(batches_indexes, [1] + shape.core_shape_list)
    batches_indexes = tf.cast(batches_indexes, int_data_type)

    axis_ind = shape.core_rank_int + 1

    Ia = tf.gather_nd(images, tf.stack([batches_indexes, y0, x0], axis_ind))
    Ib = tf.gather_nd(images, tf.stack([batches_indexes, y1, x0], axis_ind))
    Ic = tf.gather_nd(images, tf.stack([batches_indexes, y0, x1], axis_ind))
    Id = tf.gather_nd(images, tf.stack([batches_indexes, y1, x1], axis_ind))

    wa = tf.expand_dims(wa, axis=axis_ind)
    wb = tf.expand_dims(wb, axis=axis_ind)
    wc = tf.expand_dims(wc, axis=axis_ind)
    wd = tf.expand_dims(wd, axis=axis_ind)

    # Trilinear interpolation step
    return tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])


def _trilinear_interpolation(
    images: tf.Tensor,
    sampling_grid: tf.Tensor,
    shape: ShapeBreakDown,
    int_data_type=tf.int32,
) -> tf.Tensor:
    """
    Trilinear interpolation, from sampling points.

    Arguments
    ---------
    - `images`: batch of 3D images to interpolate.
    - `sampling_grid`:
    Returns:
        image transformed according to the displacement vector field
    """
    max_y = shape.height_int - 1
    max_x = shape.width_int - 1
    max_z = shape.height_int - 1

    # if tf.math.reduce_all(p == 0):
    #     return img

    x = sampling_grid[:, :, :, :, 0]
    y = sampling_grid[:, :, :, :, 1]
    z = sampling_grid[:, :, :, :, 2]

    x0 = tf.floor(x)
    x1 = x0 + 1
    y0 = tf.floor(y)
    y1 = y0 + 1
    z0 = tf.floor(z)
    z1 = z0 + 1

    # calculate deltas
    wa = (x1 - x) * (y1 - y) * (z1 - z)
    wb = (x1 - x) * (y1 - y) * (z - z0)
    wc = (x1 - x) * (y - y0) * (z1 - z)
    wd = (x1 - x) * (y - y0) * (z - z0)
    we = (x - x0) * (y1 - y) * (z1 - z)
    wf = (x - x0) * (y1 - y) * (z - z0)
    wg = (x - x0) * (y - y0) * (z1 - z)
    wh = (x - x0) * (y - y0) * (z - z0)

    # note that casting should be done after weights/deltas calculation, to
    # avoid zeros in deltas due to rounding, since they will cause artifacts
    x0 = tf.cast(x0, int_data_type)
    x1 = tf.cast(x1, int_data_type)
    y0 = tf.cast(y0, int_data_type)
    y1 = tf.cast(y1, int_data_type)
    z0 = tf.cast(z0, int_data_type)
    z1 = tf.cast(z1, int_data_type)

    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)
    z0 = tf.clip_by_value(z0, 0, max_z)
    z1 = tf.clip_by_value(z1, 0, max_z)

    # === generate batches only indexes, to be blended with other indexes
    batches_indexes = tf.range(0, shape.batch_size)
    batches_indexes = tf.reshape(
        batches_indexes, [shape.batch_size_int] + [1] * shape.core_rank_int
    )
    batches_indexes = tf.tile(batches_indexes, [1] + shape.core_shape_list)
    batches_indexes = tf.cast(batches_indexes, int_data_type)

    axis_ind = shape.core_rank_int + 1

    Ia = tf.gather_nd(images, tf.stack([batches_indexes, y0, x0, z0], axis_ind))
    Ib = tf.gather_nd(images, tf.stack([batches_indexes, y0, x0, z1], axis_ind))
    Ic = tf.gather_nd(images, tf.stack([batches_indexes, y1, x0, z0], axis_ind))
    Id = tf.gather_nd(images, tf.stack([batches_indexes, y1, x0, z1], axis_ind))
    Ie = tf.gather_nd(images, tf.stack([batches_indexes, y0, x1, z0], axis_ind))
    If = tf.gather_nd(images, tf.stack([batches_indexes, y0, x1, z1], axis_ind))
    Ig = tf.gather_nd(images, tf.stack([batches_indexes, y1, x1, z0], axis_ind))
    Ih = tf.gather_nd(images, tf.stack([batches_indexes, y1, x1, z1], axis_ind))

    wa = tf.expand_dims(wa, axis=axis_ind)
    wb = tf.expand_dims(wb, axis=axis_ind)
    wc = tf.expand_dims(wc, axis=axis_ind)
    wd = tf.expand_dims(wd, axis=axis_ind)
    we = tf.expand_dims(we, axis=axis_ind)
    wf = tf.expand_dims(wf, axis=axis_ind)
    wg = tf.expand_dims(wg, axis=axis_ind)
    wh = tf.expand_dims(wh, axis=axis_ind)

    # Trilinear interpolation step
    return tf.add_n(
        [wa * Ia, wb * Ib, wc * Ic, wd * Id, we * Ie, wf * If, wg * Ig, wh * Ih]
    )
