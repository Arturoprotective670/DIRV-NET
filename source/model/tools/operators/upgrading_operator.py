from typing import List
import tensorflow as tf

from source.model.tools.operators.resizing_operator import (
    ResizingOperator,
    ResizingMethods,
)


class UpgradingOperator:
    """
    Purpose
    -------
    - Should be used to scale vector flow fields.
    - Scales up/down batch 2D/3D fields by factor of 2, including their
      magnitude.
    - Uses various resizing methods.
    - Creates standard pyramids of batch 2D/3D images.

    Notes
    -----
    - This name is a replacement of `ScalingOperator` used in text.

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(
        self,
        images_rank: int,
        up_resizing_method: ResizingMethods,
        down_resizing_method: ResizingMethods,
    ) -> None:
        self._resizer_up = ResizingOperator(images_rank, up_resizing_method)
        self._resizer_down = ResizingOperator(images_rank, down_resizing_method)

        self._rank = images_rank
        self._scale: int = 2
        self._scale_factors = [self._scale] * images_rank

    def upgrade_up(self, data: tf.Tensor) -> tf.Tensor:
        return self._scale * self._resizer_up.resize_up(data, self._scale_factors)

    def upgrade_down(self, data: tf.Tensor) -> tf.Tensor:
        return self._resizer_down.resize_down(data, self._scale_factors) / self._scale

    def create_pyramid(
        self, batch_fields: tf.Tensor, levels_count: int
    ) -> List[tf.Tensor]:
        """
        Creates a pyramid of batch 2D/3D vector fields.

        Arguments
        ---------
        - `batch_fields`: Vector fields to create pyramid from, should have the
        shape `[batch_size, H, W, vector_components]` or `[batch_size, H, W, D,
        vector_components]`. - `levels_count`: The number of pyramid levels,
        this number includes the original image.

        Note:
        -----
        - The pyramid levels are returned from smaller to larger one.
        """

        levels = self._resizer_down.create_pyramid(
            batch_fields, levels_count, self._scale_factors
        )

        for level_index, level in enumerate(levels):
            levels[level_index] = level / self._scale ** (
                levels_count - 1 - level_index
            )

        return levels
