from typing import List, Tuple, Union

import tensorflow as tf
import numpy as np

from source.model.tools.operators.resizing_operator import (
    ResizingOperator,
    ResizingMethods,
)
from source.model.tools.operators.convolution_operator import ConvolutionOperator
from source.model.tools.shape_break_down import ShapeBreakDown

from source.framework.tools.general import gaussian_kernel


class NonRigidRandomFields:
    """
    Purpose
    -------
    - Generates random, non-rigid (local), smoothed vector fields.
    - Supports batch 2D/3D field generating.
    - Fields complexity (amount of variations) can be controlled, and randomized.
    - Fields magnitude can be controlled, and randomized.

    Notes
    -----
    - This a non-naive random field generating, since generating them just from
    a couple of random numbers, then interpolating them, will introduce various
    patterns/artifacts, that are not suitable for machine learning.

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(
        self,
        core_rank: int,
        resizing_method: ResizingMethods,
        smoothing_kernels_config: List[Tuple[int, float]],
        random_seed: int,
    ) -> None:
        """
        Arguments
        ---------
        - `core_rank`: the rank of the desired vector fields, used to initialize
        the class.
        - `resizing_method`: the method used to scale up the vector fields.
        - `smoothing_kernels_config`: the list of tuples containing the size of
        the Gaussian kernels and its sigmas that will be used for smoothing
        (applied in the same order).
        - `random_seed`: the seed of the random number generator.
        """
        self._core_rank = core_rank

        self._make_like_shape = ResizingOperator(
            core_rank,
            resizing_method,
        ).make_like_shape

        self._conv_op = ConvolutionOperator(core_rank).convolve

        self._kernels = []
        self._smoothing_margin: int = 0

        # === crete kernels
        for options in smoothing_kernels_config:
            kernel = gaussian_kernel(core_rank, size=options[0], sigma=options[1])
            # kernels for convolution should have in/out channels count last dimensions
            kernel = tf.expand_dims(kernel, axis=-1)
            kernel = tf.expand_dims(kernel, axis=-1)

            if options[0] > self._smoothing_margin:
                self._smoothing_margin = options[0]

            self._kernels.append(kernel)

        # === round to even number (see the reason below)
        if self._smoothing_margin % 2 != 0:
            self._smoothing_margin += 1

        tf.random.set_seed(random_seed)

    def generate_randomized(
        self,
        initial_seed_range: Tuple[int, int],
        max_magnitude_range: Tuple[float, float],
        target_batch_shape: ShapeBreakDown,
        random_generator: np.random.Generator = np.random.default_rng(),
    ) -> tf.Tensor:
        """
        Generates a smooth 2D/3D random vector field, with randomized attributes.

        Arguments
        ---------
        - `initial_seed_range`: the range of dimensions of the initial seed tensor
        that will be scaled up to the `target_batch_shape`.
        - `max_magnitude_range`: the range of the maximum magnitude of the
        generated vector field.
        - `target_batch_shape`: the shape object of the targeted vector field.
        - `random_generator`: the numpy random number generator to use.

        Returns
        -------
        - A 2D/3D tensor of shape: `[batch_size, height, width, (depth), channels, vector_component]`
        where `vector_component` represents the `X, Y, ...` components of the
        random flow field.

        Notes
        -----
        - The bigger `initial_seed_range`, the more complex the vector field will
        be (i.e. more times it will go through zero).
        """
        initial_seed_size = random_generator.integers(
            initial_seed_range[0],
            initial_seed_range[1],
            target_batch_shape.core_rank_int,
        )

        max_magnitude = random_generator.uniform(
            max_magnitude_range[0], max_magnitude_range[1], 1
        )

        return self.generate(
            tuple(initial_seed_size), max_magnitude[0], target_batch_shape
        )

    def generate(
        self,
        initial_seed_size: Union[Tuple[int, int], Tuple[int, int, int]],
        max_magnitude: float,
        target_batch_shape: ShapeBreakDown,
    ) -> tf.Tensor:
        """
        Generates a smooth 2D/3D random vector field.

        Arguments
        ---------
        - `initial_seed_size`: the dimensions of the initial seed tensor that
        will be scaled up to the `target_batch_shape`.
        - `max_magnitude`: the maximum magnitude of the generated vector field.
        - `target_batch_shape`: the shape object of the targeted vector field.

        Returns
        -------
        - A 2D/3D tensor of shape: `[batch_size, height, width, (depth), channels, vector_component]`
        where `vector_component` represents the `X, Y, ...` components of the
        random flow field.

        Notes
        -----
        - The bigger `initial_seed_size`, the more complex the vector field will
        be (i.e. more times it will go through zero).
        - The bigger `initial_seed_size`, the bigger Gaussian kernels to make it
        well smoothed.
        - The bigger `initial_seed_size`, the more likely `max_magnitude` will
        be hit.
        """
        # === define the shape of the tensor
        shape = (
            [target_batch_shape.batch_size_int]
            + list(initial_seed_size)
            + [target_batch_shape.core_rank_int]  # for x and y and z if any
        )

        # === generate a random tensor
        field = tf.random.uniform(shape, minval=-max_magnitude, maxval=max_magnitude)

        original_magnitude_max = tf.reduce_max(tf.abs(field))

        # === smoothing
        # we need to smooth each field component separately, so applying the usual
        # convolution in TF will not work, and although TF has
        # tf.nn.depthwise_conv2d function for that, this one has no 3D
        # equivalent, so we will apply usual convolution on each field
        # component separately as a workaround, then stack them back
        field = self._grouped_convolution(
            field,
            self._kernels,
            strides=[1] * target_batch_shape.batch_rank_int,
            padding="SAME",
        )

        # === smoothing creates faded pixels on the borders, so we will resize
        # it to a bigger size (bigger by the biggest kernel size), then we crop
        # the artifacts.

        smoothed_target_batch_shape = (
            [target_batch_shape.batch_size_int]
            + [i + self._smoothing_margin for i in target_batch_shape.core_shape_list]
            + [target_batch_shape.core_rank_int]
        )
        margined_shape = ShapeBreakDown(
            tf.convert_to_tensor(smoothed_target_batch_shape), True
        )

        # === resize
        field = self._make_like_shape(field, margined_shape)

        # crop margins
        margin: int = int(self._smoothing_margin // 2)

        if target_batch_shape.core_rank_int == 3:
            field = field[
                :,
                margin:-margin,
                margin:-margin,
                margin:-margin,
                :,
            ]
        elif target_batch_shape.core_rank_int == 2:
            field = field[:, margin:-margin, margin:-margin, :]
        else:
            raise ValueError("Batch rank must be 2 or 3")

        # === restore magnitude of the field
        # since smoothing reduces the magnitude of the field, so we scale it back
        magnitude_scale_factor = original_magnitude_max / tf.reduce_max(tf.abs(field))

        return magnitude_scale_factor * field

    def _grouped_convolution(
        self,
        tensor: tf.Tensor,
        kernels: List[tf.Tensor],
        strides: List[int],
        padding: str,
    ) -> tf.Tensor:
        """
        Applies a grouped convolution (i.e. applies the kernels on each tensor
        slice separately), for 2D/3D kernels (the name is borrowed from PyTorch).
        """
        components = tf.unstack(tensor, axis=-1)

        # === separate components, and restore the lost dim. to make the
        # convolution work
        for component_index, component in enumerate(components):
            components[component_index] = tf.expand_dims(component, axis=-1)

        for component_index, _ in enumerate(components):
            for kernel in kernels:
                components[component_index] = self._conv_op(
                    components[component_index],
                    kernel,
                    strides=strides,
                    padding=padding,
                )

        # === remove the extra dim. to stack them again
        for component_index, component in enumerate(components):
            components[component_index] = tf.squeeze(component, axis=-1)

        return tf.stack(components, axis=-1)
