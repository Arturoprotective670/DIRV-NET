from enum import IntEnum
import tensorflow as tf
import numpy as np
from typing import List

from source.model.tools.b_splines_interpolation import (
    batch_b_splines_interpolation,
    batch_coordinates_map,
)
from source.model.tools.shape_break_down import ShapeBreakDown
from source.model.tools.operators.convolution_operator import ConvolutionOperator


class ResizingMethods(IntEnum):
    """
    An enum that determines the resizing methods.
    """

    LINEAR_SMOOTHED = 0
    NEAREST_NEIGHBOR = 1
    B_SPLINES = 2
    SMOOTHED_NEAREST_NEIGHBOR = 3


class ResizingOperator:
    """
    Purpose
    -------
    - Support up/down sampling a 4D or 5D Tensor of 2D/3D images.
    - Various methods for resizing.
    - Allows fractional scaling factors for some of the resizing methods.
    - Uses 2D/3D convolution and B-Splines to archive that.
    - Takes care of smoothing and close-to-borders artifacts.
    - Creates pyramids of images batches.
    - Can auto-decide what to do when asked to make on tensor like another.

    Notes
    -----
    - This name is a replacement of `ScalingOperator` used in text.
    - As of 2024 year, there is no available libs or functions to down sample
    volumetric (3D) images on internet, neither within `Pytorch`, nor in
    `Tensorflow`.
    - Some volumetric up-sampling is provided in
    `tf.keras.backend.resize_volumes` and `tf.keras.layers.UpSampling3D`,
    however, down sampling is not supported.
    - `Tensorflow graphics`, provides `downsampling` function (more precisely,
    pyramid generation) of 2D images only, and uses strided 2D convolution for
    that, as can be seen here:
    https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/image/pyramid.py#L153-L178
    this means that no custom scaling factor can be specified (only half size is
    possible).
    - Here, to keep things within Tensorflow realm, we will implement a 2D/3D
    resizing, by using nearest neighbor (a.k.a up/down sampling), convolution
    with an approximated Gaussian smoothing (by Binomial kernels of 3'rd order),
    and B-Splines (since the project already contains the needed functions for
    B-Splines), with custom integer scaling factors.

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(
        self,
        images_rank: int,
        method: ResizingMethods = ResizingMethods.LINEAR_SMOOTHED,
        spline_degree=1,
    ) -> None:
        self._method = method
        self._spline_degree = spline_degree

        convolver = ConvolutionOperator(images_rank)
        self._conv = convolver.convolve
        self._conv_transpose = convolver.convolve_transpose

        self._smooth_ahead = False

        if method == ResizingMethods.B_SPLINES:
            self._resize_up = self._spline_resize
            self._resize_down = self._spline_resize
            self._coordinate_map = None
            self._base_shape_of_map = None

        elif method == ResizingMethods.LINEAR_SMOOTHED:
            self._resize_down = self._linear_resize_down
            self._resize_up = self._linear_resize_up

        elif method == ResizingMethods.NEAREST_NEIGHBOR:
            self._resize_down = self._downsample
            self._resize_up = self._upsample

        elif method == ResizingMethods.SMOOTHED_NEAREST_NEIGHBOR:
            self._resize_down = self._downsample
            self._resize_up = self._upsample
            self._smooth_ahead = True

        else:
            raise NotImplementedError

    def _spline_resize(self, batch_data: tf.Tensor, scale_factors: List):
        target_size = self._shape.core_shape_float * tf.convert_to_tensor(
            scale_factors, dtype=tf.float32
        )
        target_size = tf.round(target_size)
        target_size = tf.cast(target_size, dtype=tf.int32)

        if self._coordinate_map is None or self._base_shape_of_map != self._shape:
            self._coordinate_map = batch_coordinates_map(
                self._shape.core_shape,
                target_size,
                self._shape.batch_size,
                batch_data.dtype,
            )
            self._base_shape_of_map = self._shape

        return batch_b_splines_interpolation(
            batch_data,
            self._coordinate_map,
            self._spline_degree,
            [False] * self._shape.core_rank_int,
            tf.int32,
        )

    def _linear_resize_down(self, data: tf.Tensor, scale_factors: List[int]):
        strides = [1] + scale_factors + [1]

        kernel = self._binomial_kernel()

        data = self._pad_for_downsampling(
            data, scale_factors, tf.shape(kernel).numpy(), strides[1:-1]
        )

        return self._conv(input=data, filters=kernel, strides=strides, padding="VALID")

    def resize_down(self, data: tf.Tensor, scale_factors: List) -> tf.Tensor:
        self._shape = ShapeBreakDown(data)

        if self._method == ResizingMethods.B_SPLINES:
            scale_factors = [1 / (i + 1e-8) for i in scale_factors]
        else:
            scale_factors = [int(i) for i in scale_factors]

        return self._resize_down(data, scale_factors)

    def _binomial_kernel(self) -> tf.Tensor:
        """
        Creates a 5x5 binomial kernel, of order 3, that approximates Gaussian
        smoothing.

        Arguments
        ---------
        `rank`: `channels_count`: The number of channels of the image to filter.
        `dtype`: The type of an element in the kernel.

        Returns
        -------
        A tensor of shape `[5, 5, num_channels, num_channels]`.
        """

        # this is basically a generalization of
        # https://github.com/tensorflow/graphics/blob/98c0d63f1eb8b475070e0ae94f42386842862512/tensorflow_graphics/image/pyramid.py#L55

        # to avoid over-smoothing, if the shape is less than 1.5 of the kernel
        # shape, we will use the smaller kernel
        if max(self._shape.core_shape_list) / 2 > 5:
            order = 3
        else:
            order = 1

        if order == 3:
            kernel = np.array(
                (1.0, 4.0, 6.0, 4.0, 1.0), dtype=self._shape.data_type.as_numpy_dtype
            )
        elif order == 1:
            kernel = np.array(
                (1.0, 2.0, 1.0), dtype=self._shape.data_type.as_numpy_dtype
            )

        if self._shape.core_rank_int == 2:
            kernel = np.outer(kernel, kernel)
            kernel = kernel[:, :, np.newaxis, np.newaxis]

        elif self._shape.core_rank_int == 3:
            kernel_size = kernel.shape[0]
            kernel = np.outer(kernel, np.outer(kernel, kernel))
            kernel = kernel.reshape(kernel_size, kernel_size, kernel_size)
            kernel = kernel[:, :, :, np.newaxis, np.newaxis]

        else:
            raise NotImplementedError

        kernel /= np.sum(kernel)

        return tf.constant(kernel, dtype=self._shape.data_type) * tf.eye(
            self._shape.channels_count, dtype=self._shape.data_type
        )

    def _nearest_neighbor_downsample_kernel(self, scale_factors: List) -> tf.Tensor:
        # Create a kernel of zeros with shape (*scale_factors)
        kernel = np.zeros(
            [*scale_factors, 1, self._shape.channels_count_int], dtype=float
        )

        # Assign the center value of the kernel to 1, so it will be the only one
        # that picked among the pixels covered by the kernel
        kernel_centers = np.array(scale_factors) // 2

        if (
            self._shape.core_rank_int == 2
        ):  # replace by unpaking (works only in python 3.11+)
            kernel[kernel_centers[0], kernel_centers[1], ...] = 1.0

        elif self._shape.core_rank_int == 3:
            kernel[kernel_centers[0], kernel_centers[1], kernel_centers[2], ...] = 1.0

        else:
            raise NotImplementedError

        # convert to tensor
        return tf.constant(kernel, dtype=self._shape.data_type)

    def _smooth_batch(self, batch_images: tf.Tensor):
        strides = [1] * self._shape.batch_rank_int
        kernel = self._binomial_kernel()

        batch_images = self._pad_for_downsampling(
            batch_images,
            [1] * self._shape.core_rank_int,
            tf.shape(kernel).numpy(),
            strides[1:-1],
        )

        return self._conv(batch_images, kernel, strides=strides, padding="VALID")

    def _downsample(self, batch_images: tf.Tensor, scale_factors: List):
        if self._smooth_ahead:
            batch_images = self._smooth_batch(batch_images)

        kernel = self._nearest_neighbor_downsample_kernel(scale_factors)
        strides = [1] + scale_factors + [1]
        batch_images = self._pad_for_downsampling(
            batch_images, scale_factors, tf.shape(kernel).numpy(), strides[1:-1]
        )

        return self._conv(batch_images, kernel, strides=strides, padding="VALID")

    def _pad_for_downsampling(
        self, batch_images, scale_factors, filter_size, core_strides
    ):
        padding_list = [[0, 0]]
        core_shape = self._shape.core_shape_list

        for dim in range(self._shape.core_rank_int):
            pad = (core_shape[dim] / scale_factors[dim]) * core_strides[dim]
            pad += filter_size[dim] - core_shape[dim] - 1
            pad = int(pad)
            padding_list.append([pad // 2, (pad + 1) // 2])

        padding_list.append([0, 0])
        paddings = tf.constant(padding_list, dtype=tf.int32)

        # Pad the image with "reflection" padding
        return tf.pad(batch_images, paddings, mode="REFLECT")

    def _nearest_neighbor_upsample_kernel(self, scale_factors: List) -> tf.Tensor:
        kernel = tf.ones([*scale_factors, 1, self._shape.channels_count_int])
        return tf.constant(kernel, dtype=self._shape.data_type)

    def _upsample(self, batch_images: tf.Tensor, scale_factors: List):
        if self._smooth_ahead:
            batch_images = self._smooth_batch(batch_images)

        kernel = self._nearest_neighbor_upsample_kernel(scale_factors)
        new_images_shape = np.array(self._shape.core_shape_list) * np.array(
            scale_factors
        )

        output_shape = [
            self._shape.batch_size_int,
            *list(new_images_shape),
            self._shape.channels_count_int,
        ]
        strides = [1] + scale_factors + [1]

        return self._conv_transpose(
            batch_images, kernel, output_shape, strides=strides, padding="SAME"
        )

    def _linear_resize_up(self, data: tf.Tensor, scale_factors: List[int]):
        strides = [1] + scale_factors + [1]
        new_images_shape = self._shape.core_shape * tf.convert_to_tensor(scale_factors)

        output_shape = (
            [self._shape.batch_size_int]
            + list(new_images_shape.numpy())
            + [self._shape.channels_count_int]
        )

        kernel = 2**self._shape.core_rank_int * self._binomial_kernel()

        return self._conv_transpose(
            input=data,
            filters=kernel,
            output_shape=output_shape,
            strides=strides,
            padding="SAME",
        )

    def resize_up(self, data: tf.Tensor, scale_factors: List) -> tf.Tensor:
        self._shape = ShapeBreakDown(data)

        if self._method == ResizingMethods.B_SPLINES:
            pass
        else:
            scale_factors = [int(i) for i in scale_factors]

        return self._resize_up(data, scale_factors)

    def create_pyramid(
        self,
        batch_images: tf.Tensor,
        levels_count: int,
        scale_factors: List[int] = None,
    ) -> List[tf.Tensor]:
        """
        Creates a pyramid of batch 2D/3D images.

        Arguments
        ---------
        - `batch_images`:
        - `levels_count`: The number of pyramid levels, this number includes
        the original image.
        - `scale_factors`: A List of scale factors per image dimension.

        Note:
        -----
        - The pyramid levels are returned from smaller to larger one.
        """
        self._shape = ShapeBreakDown(batch_images)

        if scale_factors is None:
            scale_factors = [2] * self._shape.core_rank_int

        MIN_PYRAMID_SIZE = 2
        max_scale_factors = tf.convert_to_tensor(
            scale_factors, dtype=batch_images.dtype
        )
        max_scale_factors **= levels_count - 1
        if tf.reduce_any(
            self._shape.core_shape_float / max_scale_factors < MIN_PYRAMID_SIZE
        ):
            raise Exception("Too many pyramid levels for this image size.")

        levels = [batch_images]

        for _ in range(levels_count - 1):
            batch_images = self.resize_down(batch_images, scale_factors)
            levels.append(batch_images)

        levels.reverse()

        return levels

    def make_like(self, batch_images: tf.Tensor, target: tf.Tensor):
        return self.make_like_shape(batch_images, ShapeBreakDown(target))

    def make_like_shape(self, batch_images: tf.Tensor, target_shape: ShapeBreakDown):
        self._shape = ShapeBreakDown(batch_images)

        scale_factors = target_shape.core_shape_float / self._shape.core_shape_float

        if tf.reduce_all(scale_factors > 1):
            return self.resize_up(batch_images, scale_factors.numpy())
        elif tf.reduce_all(scale_factors < 1):
            return self.resize_down(batch_images, scale_factors.numpy())
        elif tf.reduce_all(scale_factors == 1):
            return batch_images  # no resizing needed
        else:
            raise NotImplementedError
