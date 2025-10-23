import tensorflow as tf

from source.model.main.pipe_line.pipe_line_data import (
    PyramidLevelData,
    VariationalUnitData,
)
from source.model.main.learnable_variables import LearnableVariables
from source.model.tools.shape_break_down import ShapeBreakDown
from source.model.tools.operators.resizing_operator import ResizingOperator
from source.model.tools.operators.warping_operator import warp_images

from source.model.main.pipe_line.variational_unit import VariationalUnit
from source.model.settings.config import Config


class PyramidLevel:
    """
    Purpose
    --------
    - Represents a pyramid level object in the DIRV-Net (see eq. 21).

    Contributors
    ------------
    - TMS-Namespace
    - Claudio Fanconi
    """

    def __init__(
        self, config: Config, learnable_variables: LearnableVariables, pyramid_level: int
    ) -> None:
        self._config = config
        self._pyramid_level = pyramid_level

        # generate pyramid's VUs list
        self._variational_units = [
            VariationalUnit(config, learnable_variables, pyramid_level, vu_index).push
            for vu_index in range(config.unfoldings_count)
        ]

        self._interpolator = ResizingOperator(
            config.images_rank, config.displacements_to_flows_resizing_up_method
        ).make_like

        self._shape = None

        self._displacements_grid_size = None
        self._initial_displacements = None

    def _init(self, fixed_images: tf.Tensor) -> None:
        """
        We have a bench of constants, that stays the same as long as the batch
        size did not change, so here we pre-calculate them.
        """

        new_shape = ShapeBreakDown(fixed_images)
        if self._shape is not None and self._shape == new_shape:
            return

        self._shape = new_shape

        self._displacements_grid_size: tf.Tensor = (
            new_shape.core_shape_float
            / self._config.displacements_control_points_spacings_tf
        )

        self._displacements_grid_size = tf.cast(
            self._displacements_grid_size, dtype=self._config.int_data_type
        )

        # use pyramid level dimensions, to generate zero valued initial
        # displacements, in principle, this should be needed only for the
        # smallest pyramid level
        self._initial_displacements = tf.zeros(
            [
                new_shape.batch_size,
                *list(self._displacements_grid_size.numpy()),
                new_shape.core_rank_int,
            ],
            dtype=self._config.float_data_type,
        )

    def push(self, previous_vu_data: VariationalUnitData) -> PyramidLevelData:
        """
        Pushes data through the pyramid level pipeline (see eq. 16).
        """
        self._init(previous_vu_data.fixed_images)

        pl_data = PyramidLevelData()

        for vu_index, vu in enumerate(self._variational_units):
            vu_data = vu(previous_vu_data)

            if vu_index == self._config.unfoldings_count - 1:  # last VU
                # calc and add last VU registered image
                vu_data.predicted_flow = self._interpolator(
                    vu_data.predicted_displacements, vu_data.moving_images
                )
                vu_data.registered_images, _ = warp_images(
                    vu_data.moving_images,
                    vu_data.predicted_flow,
                    self._config.fields_min_threshold,
                )

            pl_data.variational_units.append(vu_data)

            previous_vu_data = vu_data

        return pl_data
