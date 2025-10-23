"""
Purpose
-------
- Apply flow fields on images, by warping them using bilinear or trilinear
  interpolation.

References
----------
- https://meghal-darji.medium.com/implementing-bilinear-interpolation-for-image-resizing-357cbb2c2722
- https://en.wikipedia.org/wiki/Bilinear_interpolation

Notes
-----
- See a "from scratch" alternative implementation in
  `support/scripts/tools/warper_alternative.py`.

Contributors
------------
- TMS-Namespace
"""

from typing import Optional, Tuple

import tensorflow as tf

from tensorflow_graphics.math.interpolation.trilinear import interpolate as tf_trilinear
from tensorflow_addons.image import interpolate_bilinear as tf_bilinear

from source.model.tools.shape_break_down import ShapeBreakDown
from source.model.tools.mesh_generator import batch_flat_mesh_axises_list


def warp_images(
    images: tf.Tensor, flow_fields: tf.Tensor, fields_threshold: float = 1e-3
) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    """
    Transforms/warps an images batch according to the provided flow field, by
    using Bilinear/Trilinear interpolation.

    Arguments
    ---------
    - `images_batch`: the source images batch to interpolate.
    - `flow_fields`: the batch of the flow fields.
    - `fields_threshold`: the threshold of the flow fields, below which they
      will be considered as absent.

    Returns
    -------
    - A tensor of warped images batch according to the provided flow vector
      fields.
    - A tensor of the used sampling points for interpolation, in a flat format,
    that been generated from the provided flow fields.

    Notes
    -----
    - This function performs a 'pull' (or backward) resampling, this why it
    needs inverse transformations, or flaw fields.
    - It uses the so usually called "nearest" mode, for out of boundaries
    interpolation, i.e. any out of image boundaries pixels, will be filled with
    the nearest pixel.
    -  if flow field is too small, there is no sense of applying it, experiments
    showed that fields_threshold = 1e-3 is a good threshold to cause less than 1
    in pixel intensity change after warping.
    """

    if tf.math.reduce_max(tf.math.abs(flow_fields)).numpy() <= fields_threshold:
        return images, None

    input_shape = ShapeBreakDown(images)

    flat_mesh_axises, _ = batch_flat_mesh_axises_list(
        input_shape.core_shape, input_shape.batch_size_int
    )

    # we need flat sampling points for the TF interpolation functions
    original_flat_mesh = tf.concat(flat_mesh_axises, axis=-1)
    flat_flows = tf.reshape(
        flow_fields, [input_shape.batch_size_int, -1, input_shape.core_rank_int]
    )

    flat_transformation_grid = original_flat_mesh + flat_flows

    # our flows and mesh contains sampling points coordinates as x,y, but TF
    # interpolation function expects y,x, so we need to swap axes, we do this
    # for images them selves instead of doing that for sampling pints, just
    # because it is easier (flows will revert them back during interpolation).
    if input_shape.core_rank_int == 2:
        images = tf.transpose(images, [0, 2, 1, 3])
        warped_images = tf_bilinear(images, query_points=flat_transformation_grid)
    elif input_shape.core_rank_int == 3:
        images = tf.transpose(images, [0, 2, 1, 3, 4])
        warped_images = tf_trilinear(images, sampling_points=flat_transformation_grid)
    else:
        raise NotImplementedError

    # go back from flat to original shape
    warped_images = tf.reshape(warped_images, shape=input_shape.batch_shape)
    flat_transformation_grid = tf.reshape(
        flat_transformation_grid, shape=input_shape.flow_filed_shape_list
    )

    return warped_images, flat_transformation_grid
