"""
Purpose
-------
- Generate 2D/3D affine transformation matrices.
- Generate random affine transformations.
- Generate flow fields from those transformation matrixes.

References
----------
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html
- https://people.computing.clemson.edu/~dhouse/courses/401/notes/affines-matrices.pdf
- https://en.wikipedia.org/wiki/Affine_transformation
- https://www.geogebra.org/m/Fq8zyEgS

Contributors
------------
- TMS-Namespace
"""

from typing import Tuple, Union
import tensorflow as tf
import numpy as np
from numpy import float64
from numpy.typing import NDArray

from source.model.tools.mesh_generator import batch_flat_mesh_axises_list


def affine_matrix_2d(
    translation: Tuple[float, float] = (0, 0),
    rotation: float = 0.0,
    shear: Tuple[float, float] = (0, 0),
    scale: Tuple[float, float] = (1, 1),
    origin_changer: NDArray[float64] = np.eye(3),
    transformations_sequence: str = "tirhso",
) -> NDArray[float64]:
    """
    Generates `2D` homogenies affine transformation matrix.

    Arguments
    ---------
    - `translation`: The `(x,y)` translations to apply.
    - `rotation`: The rotation angle in radians.
    - `shear`: The `(x,y)` shear amount to apply.
    - `origin_changer`: a homogenies matrix to change the origin of coordinates.
    - `transformations_sequence`: a sequence of the lattes from the set
    `t, i, r, h, s, o`, that determines in which order transformations will be
    applied on the (reads from right to left), where:
        -  `t`: stands for translations.
        -  `i`: stands for inverted origin changing matrix.
        -  `r`: stands for rotation.
        -  `h`: stands for shear.
        -  `s`: stands for scaling.
        -  `o`: stands for origin changing matrix.

    Returns:
    --------
    - A `3x3` array, where the last row is the homogeneity row (i.e. consists of `[0, 0, 1]`).

    Notes:
    ------
    - The transformation assumes origin of coordinates in the middle.
    - It is possible to use this same function to generate `origin_changer` matrix.
    - `transformations_sequence` should be set carefully, since for example, if
    `origin_changer` matrix contains axis inversion, it may make translations
    to happen in the opposite direction, in case of `i` is applied at the sequence
    end, instead of being applied before the translation (as in the default value
    of `transformations_sequence`).
    """

    matrix: NDArray[float64] = np.eye(3)
    current_trans_matrix: NDArray[float64] = np.eye(3)

    for trans_type in reversed(transformations_sequence):
        if trans_type == "t":  # Translation
            current_trans_matrix = np.array(
                [[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]]
            )
        elif trans_type == "r":  # Rotation
            current_trans_matrix = np.array(
                [
                    [np.cos(rotation), -np.sin(rotation), 0],
                    [np.sin(rotation), np.cos(rotation), 0],
                    [0, 0, 1],
                ]
            )
        elif trans_type == "h":  # Shear
            current_trans_matrix = np.array(
                [[1, shear[0], 0], [shear[1], 1, 0], [0, 0, 1]]
            )
        elif trans_type == "s":  # Scale
            current_trans_matrix = np.array(
                [[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]]
            )
        elif trans_type == "o":  # origin changer
            current_trans_matrix = origin_changer
        elif trans_type == "i":  # inverse origin changer
            current_trans_matrix = np.linalg.inv(origin_changer)
        else:
            raise Exception("Unknown transformation type in the sequence.")

        matrix = np.matmul(current_trans_matrix, matrix)

    return matrix


def affine_matrix_3d(
    translation: Tuple[float, float, float] = (0, 0, 0),
    rotation: Tuple[float, float, float] = (0, 0, 0),
    shear: Tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0),
    scale: Tuple[float, float, float] = (1, 1, 1),
    origin_changer: NDArray = np.eye(4),
    transformations_sequence: str = "tirhso",
) -> NDArray[np.float32]:
    """
    Generates `3D` homogenous affine transformation matrix.

    Arguments
    ---------
    - `translation`: The `(x, y, z)` translations to apply.
    - `rotation`: The rotation angles in radians around the `(x, y, z)` axes.
    - `shear`: The `(xy, xz, yx, yz, zx, zy)` shear amounts to apply.
    - `scale`: The `(x, y, z)` scaling factors.
    - `origin_changer`: A homogeneous matrix to change the origin of coordinates.
    - `transformations_sequence`: A sequence of the latest from the set
    `t, i, r, h, s, o`, that determines in which order transformations will be
    applied on the (reads from right to left), where:
        -  `t`: stands for translations.
        -  `i`: stands for inverted origin changing matrix.
        -  `r`: stands for rotation.
        -  `h`: stands for shear.
        -  `s`: stands for scaling.
        -  `o`: stands for origin changing matrix.

    Returns
    -------
    - A `4x4` array, where the last row is the homogeneity row (i.e. consists of `[0, 0, 0, 1]`).

    Notes
    -----
    - The transformation assumes origin of coordinates in the middle.
    - It is possible to use this same function to generate `origin_changer` matrix.
    - `transformations_sequence` should be set carefully, since for example, if
    `origin_changer` matrix contains axis inversion, it may make translations to
    happen in the opposite direction, in case of `i` is applied at the sequence
    end, instead of being applied before the translation (as in the default
    value of `transformations_sequence`).
    """

    matrix = np.eye(4)
    current_trans_matrix = np.eye(4)

    for trans_type in reversed(transformations_sequence):
        if trans_type == "t":  # Translation
            current_trans_matrix = np.array(
                [
                    [1, 0, 0, translation[0]],
                    [0, 1, 0, translation[1]],
                    [0, 0, 1, translation[2]],
                    [0, 0, 0, 1],
                ]
            )
        elif trans_type == "r":  # Rotation
            rot_x = np.array(
                [
                    [1, 0, 0, 0],
                    [0, np.cos(rotation[0]), -np.sin(rotation[0]), 0],
                    [0, np.sin(rotation[0]), np.cos(rotation[0]), 0],
                    [0, 0, 0, 1],
                ]
            )

            rot_y = np.array(
                [
                    [np.cos(rotation[1]), 0, np.sin(rotation[1]), 0],
                    [0, 1, 0, 0],
                    [-np.sin(rotation[1]), 0, np.cos(rotation[1]), 0],
                    [0, 0, 0, 1],
                ]
            )

            rot_z = np.array(
                [
                    [np.cos(rotation[2]), -np.sin(rotation[2]), 0, 0],
                    [np.sin(rotation[2]), np.cos(rotation[2]), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )

            current_trans_matrix = np.matmul(np.matmul(rot_x, rot_y), rot_z)

        elif trans_type == "h":  # Shear
            shear_matrix = np.array(
                [
                    [1, shear[0], shear[1], 0],
                    [shear[2], 1, shear[3], 0],
                    [shear[4], shear[5], 1, 0],
                    [0, 0, 0, 1],
                ]
            )

            current_trans_matrix = shear_matrix

        elif trans_type == "s":  # Scale
            current_trans_matrix = np.array(
                [
                    [scale[0], 0, 0, 0],
                    [0, scale[1], 0, 0],
                    [0, 0, scale[2], 0],
                    [0, 0, 0, 1],
                ]
            )
        elif trans_type == "o":  # origin changer
            current_trans_matrix = origin_changer
        elif trans_type == "i":  # inverse origin changer
            current_trans_matrix = np.linalg.inv(origin_changer)
        else:
            raise ValueError("Unknown transformation type in the sequence.")

        matrix = np.matmul(current_trans_matrix, matrix)

    return matrix


def image_affine_matrix_2d(
    image_size: Tuple[int, int],
    translation: Tuple[float, float] = (0, 0),
    rotation: float = 0.0,
    shear: Tuple[float, float] = (0, 0),
    scale: Tuple[float, float] = (1, 1),
    transformations_sequence: str = "rhs",
) -> NDArray[float64]:
    """
    Generates `2D` homogenies affine transformation matrix, that can be applied on
    images (i.e. it considered that coordinates origin is in the upper left corner).

    Arguments
    ---------
    - `image_size`: a tuple that defines the target image dimensions (width, height).
    - `translation`: The `(x,y)` translations to apply.
    - `rotation`: The rotation angle in radians.
    - `shear`: The `(x,y)` shear amount to apply.
    - `scale`: The `(x, y)` scaling factors.
    - `transformations_sequence`: a sequence of the lattes from the set
    `r, h, s`, that determines in which order transformations will be
    applied on the (reads from right to left), where:
        -  `r`: stands for rotation.
        -  `h`: stands for shear.
        -  `s`: stands for scaling.

    Returns:
    --------
    - A `3x3` array, where the last row is the homogeneity row
    (i.e. consists of `[0, 0, 1]`).
    """
    origin_changer: NDArray[float64] = affine_matrix_2d(
        translation=(-image_size[0] / 2, image_size[1] / 2),
        scale=(1.0, -1.0),
        transformations_sequence="ts",
    )

    return affine_matrix_2d(
        translation,
        rotation,
        shear,
        scale,
        origin_changer,
        transformations_sequence="ti" + transformations_sequence + "o",
    )


def image_affine_matrix_3d(
    image_size: Tuple[int, int, int],
    translation: Tuple[float, float, float] = (0, 0, 0),
    rotation: Tuple[float, float, float] = (0, 0, 0),
    shear: Tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0),
    scale: Tuple[float, float, float] = (1, 1, 1),
    transformations_sequence: str = "rhs",
) -> NDArray[float64]:
    """
    Generates `3D` homogenies affine transformation matrix, that can be applied on
    images (i.e. it considered that coordinates origin is in the upper left corner).

    Arguments
    ---------
    - `image_size`: a tuple that defines the target image dimensions (width, height, depth).
    - `translation`: The `(x,y,z)` translations to apply.
    - `rotation`: The rotation angles in radians around the `(x, y, z)` axes.
    - `shear`: The `(xy, xz, yx, yz, zx, zy)` shear amounts to apply.
    - `scale`: The `(x, y, z)` scaling factors.
    - `transformations_sequence`: a sequence of the lattes from the set
    `r, h, s`, that determines in which order transformations will be
    applied on the (reads from right to left), where:
        -  `r`: stands for rotation.
        -  `h`: stands for shear.
        -  `s`: stands for scaling.

    Returns:
    --------
    - A `3x3` or `4x4` array, where the last row is the homogeneity row
    (i.e. consists of `[0, 0, 1]` or `[0, 0, 0, 1]`).

    Notes:
    ------
    - The count of the `image_size` parameter, will determine the output size.
    """
    origin_changer: NDArray[float64] = affine_matrix_3d(
        translation=(-image_size[0] / 2, image_size[1] / 2, image_size[2] / 2),
        scale=(1.0, -1.0, -1.0),
        transformations_sequence="ts",
    )

    return affine_matrix_3d(
        translation,
        rotation,
        shear,
        scale,
        origin_changer,
        transformations_sequence="ti" + transformations_sequence + "o",
    )


def random_image_affine_matrix(
    image_size: Union[Tuple[int, int], Tuple[int, int, int]],
    origin_shift_range: Tuple[int, int],
    translations_range: Tuple[float, float],
    rotations_range: Tuple[float, float],
    shears_range: Tuple[float, float],
    scales_ranges: Tuple[Tuple[float, float], Tuple[float, float]],
    random_generator: np.random.Generator = np.random.default_rng(),
) -> NDArray[float64]:
    """
    Generates `2D` or `3D` random homogenies affine transformation matrix, that
    can be applied on images (i.e. it considered that coordinates origin is in
    the upper left corner).

    Arguments
    ---------
    - `image_size`: a tuple that defines the target image dimensions (width, height),
    or (width, height, depth). It will also define the rank of the output.
    - `origin_shift_range`: The range where the origin of affine
    transformations should be randomly shifted, this will randomize the place
    where rotations, shears and scalings (only) will be applied, relative to
    the image middle.
    - `translations_range`: The range where random translations can take place.
    - `rotations_range`: The range where random rotations can take place.
    - `shears_range`: The range where random shear can take place.
    - `scales_range`: The range where random scaling can take place, consists of
    two tuples, so that positive and negative ranges can be specified, to avoid
    having a range that goes through zero.
    - `random_generator`: a numpy random generator to use for random number
    generation.

    Returns
    -------
    - A `3x3` or `4x4` array, where the last row is the homogeneity row.
    """

    image_rank = len(image_size)

    translations = random_generator.uniform(
        translations_range[0], translations_range[1], image_rank
    )
    shears = random_generator.uniform(
        shears_range[0], shears_range[1], 2 if image_rank == 2 else 6
    )
    rotations = random_generator.uniform(
        rotations_range[0], rotations_range[1], 1 if image_rank == 2 else 3
    )

    scales_range = scales_ranges[random_generator.choice([0, 1])]
    scales = random_generator.uniform(scales_range[0], scales_range[1], image_rank)

    # to shift  trans. origin, we just change randomly image size
    # we multiply by 2, since we divide by 2 when we find the image middle
    image_size = [
        image_size[i]
        + 2 * random_generator.uniform(origin_shift_range[0], origin_shift_range[1])
        for i in range(image_rank)
    ]

    trans_sequence = np.array(list("rhs"))
    random_generator.shuffle(trans_sequence)

    if image_rank == 2:
        return image_affine_matrix_2d(
            tuple(image_size),
            tuple(translations),
            rotations[0],
            tuple(shears),
            tuple(scales),
            "".join(trans_sequence),
        )
    elif image_rank == 3:
        return image_affine_matrix_3d(
            tuple(image_size),
            tuple(translations),
            tuple(rotations),
            tuple(shears),
            tuple(scales),
            "".join(trans_sequence),
        )
    else:
        raise ValueError("Unsupported rank.")


def batch_random_image_affine_matrices(
    image_size: Union[Tuple[int, int], Tuple[int, int, int]],
    origin_shift_range: Tuple[int, int],
    translations_range: Tuple[float, float],
    rotations_range: Tuple[float, float],
    shears_range: Tuple[float, float],
    scales_ranges: Tuple[Tuple[float, float], Tuple[float, float]],
    batch_size: int,
    inverse: bool = True,
    random_generator: np.random.Generator = np.random.default_rng(),
) -> tf.Tensor:
    """
    Generates a batch of `2D` or `3D` random homogenies affine transformation
    matrices, that can be applied on images (i.e. it considered that coordinates origin is in
    the upper left corner).

    Arguments
    ---------
    - `image_size`: a tuple that defines the target image dimensions (width, height),
    or (width, height, depth). It will also define the rank of the output.
    - `origin_shift_range`: The range where the origin of affine
    transformations should be randomly shifted, this will randomize the place
    where rotations, shears and scalings (only) will be applied, relative to
    the image middle.
    - `translations_range`: The range where random translations can take place.
    - `rotations_range`: The range where random rotations can take place.
    - `shears_range`: The range where random shear can take place.
    - `scales_range`: The range where random scaling can take place, consists of
    two tuples, so that positive and negative ranges can be specified, to avoid
    having a range that goes through zero.
    - `batch_size`: the output batch size, or the count of the generated matrices.
    - `inverse`: if `True` it generates inverse affine matrices batch.
    - `random_generator`: a numpy random generator to use for random number
    generation.

    Returns:
    --------
    - A tensor of homogenies affine transformation matrices of the
    shape `[batch_size, 3 or 4, 3 or 4]`.
    """

    matrices = []

    for _ in range(batch_size):
        trans_mat = random_image_affine_matrix(
            image_size,
            origin_shift_range,
            translations_range,
            rotations_range,
            shears_range,
            scales_ranges,
            random_generator,
        )
        if inverse:
            matrices.append(np.linalg.inv(trans_mat))
        else:
            matrices.append(trans_mat)

    return tf.stack(matrices, axis=0)


def flow_fields_from_batch_affine_matrices(
    batch_inverse_affine_matrices: tf.Tensor,
    output_core_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    float_data_type=tf.float32,
) -> tf.Tensor:
    """
    Generates the flow fields from a batch of inverse homogenous affine matrices.

    Arguments:
    ----------
    - `batch_inverse_affine_matrices`: A `tensor` of a batch of inverse
    homogenies `2D (3x3)` or `3D (4x4)` affine matrices, that can be applied
    on images (i.e. they consider upper left corner as coordinate origin).
    - `output_core_shape`: A list of integers that specifies the output `core shape`.
    - `float_data_type`: Output data type.

    Returns:
    --------
    - `flow_field`: A tensor of the shape `[batch_size, Y, X, ..., vector_component]`,
    where `vector_component` represents the `X, Y, ...` components of the flow field.
    """

    rank = len(output_core_shape)
    batch_size = tf.shape(batch_inverse_affine_matrices)[0].numpy()

    flat_mesh_axises, mesh_grid = batch_flat_mesh_axises_list(
        output_core_shape, batch_size
    )

    # add homogenous row to the mesh, and concat by dim. #1 in [batch_size, 1, -1]
    # so that it will be compatible with affine matrix dims
    ones = tf.ones_like(flat_mesh_axises[0])
    homogenous_flat_mesh = tf.concat(flat_mesh_axises + [ones], axis=-1)

    # affine mesh transform
    batch_inverse_affine_matrices = tf.cast(
        batch_inverse_affine_matrices, dtype=float_data_type
    )

    # to be able to multiply affine matrices of shape [batch_size, 3 or 4, 3 or 4],
    # with the mesh of shape [batch_size, X*Y or X*Y*Z, 3 or 4] we need to do a
    # transpose to make them compatible, then transpose back
    homogenous_flat_mesh = tf.transpose(homogenous_flat_mesh, perm=[0, 2, 1])
    deformed_flat_mesh = tf.matmul(
        batch_inverse_affine_matrices, homogenous_flat_mesh
    )
    deformed_flat_mesh = tf.transpose(deformed_flat_mesh, perm=[0, 2, 1])

    # go back from flat mesh to original one, and remove last homogenous row.
    deformed_mesh = []
    for dim in range(rank):
        deformed_flat_mesh_axis = deformed_flat_mesh[:, :, dim]
        deformed_mesh_axis = tf.reshape(
            deformed_flat_mesh_axis, [batch_size] + output_core_shape
        )
        deformed_mesh.append(deformed_mesh_axis)

    deformed_mesh_grid = tf.stack(deformed_mesh, axis=-1)

    flow_field = deformed_mesh_grid - mesh_grid

    return flow_field
