"""
Purpose
-------
- Functions to calc 2D/3D image gradient.

References
----------
- Adapted from the 2D case in:
https://github.com/tensorflow/tensorflow/blob/c256c071bb26e1e13b4666d1b3e229e110bc914a/tensorflow/python/ops/image_ops_impl.py#L4453-L4524

Contributors
------------
- TMS-Namespace
- Claudio Fanconi
"""

from typing import List
import tensorflow as tf

from source.model.tools.shape_break_down import ShapeBreakDown

def images_gradients(batch_images)-> List:
    """
    Calcs batch_images gradients `(dy, dx)` or `(dy, dx, dz)` for each color
    channel, via 1-step finite difference.

    Returns
    -------
    - A `List` of `Tensor`s of gradients that corresponds to each image
    dimension, each has the same size of the input.

    Notes
    ------
    - The gradient values are organized so that, for example in `2D` case,
    `[I(x+1, y) - I(x, y)]` is in location `(x, y)`. That means that dy will
    always have zeros in the last row, and dx will always have zeros in the last
    column.
    """

    shape = ShapeBreakDown(batch_images)
    gradients = []

    if shape.core_rank == 2:
        gradients.append(batch_images[:, 1:, :, :] - batch_images[:, :-1, :, :])
        gradients.append(batch_images[:, :, 1:, :] - batch_images[:, :, :-1, :])
    elif shape.core_rank == 3:
        gradients.append(batch_images[:, 1:, :, :, :] - batch_images[:, :-1, :, :, :])
        gradients.append(batch_images[:, :, 1:, :, :] - batch_images[:, :, :-1, :, :])
        gradients.append(batch_images[:, :, :, 1:, :] - batch_images[:, :, :, :-1, :])
    else:
        raise NotImplementedError

    for i in range(shape.core_rank):
        gradients[i] = _reshape(gradients[i], shape, i)

    return gradients


def _reshape(gradient: tf.Tensor, shape: ShapeBreakDown, dimension: int) -> tf.Tensor:
    target_shape = shape.batch_shape.numpy()
    target_shape[dimension + 1] = 1

    gradient = tf.concat(
        [gradient, tf.zeros(target_shape, gradient.dtype)], axis=dimension + 1
    )
    return tf.reshape(gradient, shape.batch_shape)
