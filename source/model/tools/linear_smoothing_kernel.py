"""
Purpose
-------
- Generating linear smoothing kernel for B-Spline first degree derivatives.

Reference
---------
- More info about this can be found in (pages 48-49):
    http://campar.in.tum.de/twiki/pub/Main/LorenSchwarz/thesis-070510.pdf

Contributors
------------
- TMS-Namespace
- Claudio Fanconi
"""

from typing import Union, Tuple
import numpy as np
import tensorflow as tf


def first_degree_b_spline_derivative_filter(
    control_points_spacings: Union[Tuple[int, int], Tuple[int, int, int]],
    float_data_type: tf.DType,
) -> tf.Tensor:
    """
    Initialize the filter that represents the derivative of the first degree
    B-Splines, in 2-3 dimensions.

    Arguments
    ----------
    - `control_points_spacings`: a 2 or 3 dimensional `Tensor` of the spatial
    distance between the control points that are used for B-Spline
    interpolation.
    - `float_data_type`: The `tensorflow` float data type that the output
    should use.

    Returns
    -------
    - A `Tensor` that represents a kernel of size `prod_i (2*s_i+1)`, where
    `s_i` is the i'th component of `control_points_spacings`.
    """
    # TODO: convert this to pure TF

    images_rank = len(control_points_spacings)

    if images_rank == 2:
        mesh_grids = np.mgrid[
            -control_points_spacings[0] : control_points_spacings[0] + 1,
            -control_points_spacings[1] : control_points_spacings[1] + 1,
        ]
    elif images_rank == 3:
        mesh_grids = np.mgrid[
            -control_points_spacings[0] : control_points_spacings[0] + 1,
            -control_points_spacings[1] : control_points_spacings[1] + 1,
            -control_points_spacings[2] : control_points_spacings[2] + 1,
        ]
    else:
        raise NotImplementedError

    control_points_spacings = tf.convert_to_tensor(
        control_points_spacings, dtype=float_data_type
    )

    kernel = 1.0

    for dim in range(images_rank):
        mesh_grid = np.expand_dims(mesh_grids[dim], axis=-1)
        mesh_grid = np.expand_dims(mesh_grid, axis=-1)

        mesh_grid = tf.constant(mesh_grid, dtype=float_data_type)

        kernel *= 1 - tf.abs(mesh_grid) / control_points_spacings[dim]

    return kernel
