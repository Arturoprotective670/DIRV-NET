"""
Purpose
-------
- Generate 2D/3D mesh grids as list.
- Generates batch of mesh grids.
- Generates flattened mesh grids.

Contributors
------------
- TMS-Namespace
"""

from typing import List, Tuple

import tensorflow as tf


def mesh_axis_list(shape: tf.Tensor) -> List[tf.Tensor]:
    """
    Generates a mesh grid, with coordinate origin in the upper left corner.

    Arguments
    ---------
    - `shape`: the desired shape of the mesh grid, it should be a `tensor`.

    Returns
    -------
    - a list [X,Y, ..] of coordinates mesh `tensors`.
    """

    shape_float = tf.cast(shape, dtype=tf.float32)
    rank = tf.size(shape)

    coordinates = []

    for dim in tf.range(rank):
        coordinate = tf.linspace(start=0.0, stop=shape_float[dim] - 1.0, num=shape[dim])
        coordinates.append(coordinate)

    return tf.meshgrid(*coordinates, indexing="xy")


def batch_mesh_grid(core_shape: tf.Tensor, batch_size: int) -> tf.Tensor:
    """
    Generates a batched mesh grid, with coordinate origin in the upper left
    corner.

    Arguments
    ---------
    - `core_shape`: the desired shape of the mesh grid.
    - `batch_size`: the desired batch size of the mesh.

    Returns:
    --------
    - A batch mesh grid `Tensor` of dimensions `[batch_size, *core_shape, rank]`, where last dimension.
    """
    mesh_list = mesh_axis_list(core_shape)

    # generate mesh grid for whole batch
    mesh_grid = tf.stack(mesh_list, axis=-1)

    # add batch dim
    mesh_grid = tf.expand_dims(mesh_grid, axis=0)

    # repeat by batch_size
    rank = tf.size(core_shape).numpy()
    return tf.tile(mesh_grid, [batch_size] + [1] * (rank + 1))


def batch_flat_mesh_axises_list(
    core_shape: tf.Tensor, batch_size: int
) -> Tuple[List[tf.Tensor], tf.Tensor]:
    """
    Generates a batched flat mesh grid, with coordinate origin in the upper left
    corner.

    Arguments
    ---------
    - `core_shape`: the desired shape of the mesh grid.
    - `batch_size`: the desired batch size of the mesh.

    Returns:
    --------
    - A `List` of `Tensors`, were each tensor, has the dimensions
    `[batch_size, prod(core_shape), 1]`.
    - A batch mesh grid `Tensor` of dimensions `[batch_size, *core_shape, rank]`
    (i.e. non-flattened).
    """
    rank = tf.size(core_shape)

    batch_mesh = batch_mesh_grid(core_shape, batch_size)

    batch_flat_mesh_axises = []

    for dim in range(rank):
        batch_mesh_grid_axis = batch_mesh[..., dim]
        batch_flat_mesh_axis = tf.reshape(batch_mesh_grid_axis, [batch_size, -1, 1])
        batch_flat_mesh_axises.append(batch_flat_mesh_axis)

    return batch_flat_mesh_axises, batch_mesh
