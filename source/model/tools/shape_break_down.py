from typing import List
import tensorflow as tf


class ShapeBreakDown:
    """
    Purpose
    -------
    - Extracts various dimensionality information of a tensor.
    - Provides information as Tensors as well as python native primitives.

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(
        self,
        source: tf.Tensor,
        is_source_a_shape: bool = False,
        float_data_type: tf.DType = tf.float32,
    ) -> None:
        """
        Arguments
        ---------
        - `source`: the tensor that we want to extract the shape information
          from.
        - `is_source_a_shape`: if `True`, it assumes that the provided source is
        already a "batch shape" tensor, not the shape it self.

        Notes
        -----
        - The depth property returns `1` in case of `2D` images.
        """
        if is_source_a_shape:
            self.data_type: tf.DType = float_data_type
        else:
            self.data_type: tf.DType = source.dtype

        self.batch_shape: tf.Tensor = source if is_source_a_shape else tf.shape(source)
        self.batch_shape_list: List[int] = list(self.batch_shape.numpy())
        self.batch_rank: tf.tensor = tf.size(self.batch_shape)
        self.batch_rank_int: int = self.batch_rank.numpy()

        self.batch_size: tf.Tensor = self.batch_shape[0]
        self.batch_size_int: int = self.batch_size.numpy()

        self.channels_count: tf.Tensor = self.batch_shape[-1]
        self.channels_count_int: int = self.channels_count.numpy()

        self.core_shape: tf.Tensor = self.batch_shape[1:-1]
        self.core_shape_list: List = list(self.batch_shape[1:-1].numpy())
        self.core_shape_float: tf.Tensor = tf.cast(
            self.core_shape, dtype=self.data_type
        )

        self.core_rank: tf.Tensor = tf.size(self.core_shape)
        self.core_rank_int: int = self.core_rank.numpy()

        self.image_shape = self.batch_shape[1:]

        self.height: tf.Tensor = self.core_shape[0]
        self.width: tf.Tensor = self.core_shape[1]
        self.depth: tf.Tensor = (
            tf.constant(1) if self.core_rank == 2 else self.core_shape[2]
        )

        self.height_int: int = self.height.numpy()
        self.width_int: int = self.width.numpy()
        self.depth_int: int = self.depth.numpy()

        self.flow_filed_shape_list: List[int] = (
            [self.batch_size_int] + self.core_shape_list + [self.core_rank_int]
        )
        self.flow_filed_shape = tf.convert_to_tensor(
            self.flow_filed_shape_list, dtype=self.data_type
        )

    def __eq__(self, other):
        if isinstance(other, ShapeBreakDown):
            return self.batch_shape_list == other.batch_shape_list
        return False
