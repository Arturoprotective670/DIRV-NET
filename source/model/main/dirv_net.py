import tensorflow as tf
import numpy as np
from numpy.typing import NDArray

from contextlib import nullcontext
from typing import Any, Optional, Tuple, List

from source.model.main.pipe_line.pipe_line_data import (
    PipeLineData,
    PyramidLevelData,
    VariationalUnitData,
)
from source.model.main.pipe_line.pyramid_level import PyramidLevel
from source.model.main.learnable_variables import LearnableVariables

from source.model.settings.config import Config

from source.model.tools.legit_shape import is_legit_shape
from source.model.tools.shape_break_down import ShapeBreakDown

from source.model.tools.operators.resizing_operator import ResizingOperator
from source.model.tools.operators.upgrading_operator import UpgradingOperator


class DIRVNet(tf.Module):
    """
    Purpose
    -------
    - The core variational model for image registration of DIRV-Net.

    Contributors
    ------------
    - TMS-Namespace.
    - Valery Vishnevskiy.
    - Claudio Fanconi.
    """

    def __init__(self, config: Config) -> None:
        config._init()
        self.config = config

        self._pyramid_images_generator = ResizingOperator(
            config.images_rank, config.images_resizing_method
        ).create_pyramid

        self._upgrader = UpgradingOperator(
            config.images_rank,
            config.displacements_upgrading_up_method,
            config.displacements_upgrading_down_method,
        ).upgrade_up

        self._loss: tf.Tensor = tf.convert_to_tensor(
            0.0
        )  # should be a tensor for tracking

        self.vars = LearnableVariables()
        self.vars.initialize_variables(config)

        # === generate a list of pyramid levels objects
        self._pyramid_levels = [
            PyramidLevel(self.config, self.vars, level_index)
            for level_index in range(self.config.pyramid_levels_count)
        ]

    def _forward_propagate(
        self, fixed_images: tf.Tensor, moving_images: tf.Tensor
    ) -> PipeLineData:
        """
        A help function to forward propagating through the variational network
        model.

        Arguments
        ---------
        - `fixed_images`: batch `Tensor` of the fixed images.
        - `moving_images`: batch `Tensor` of the moving images.

        Returns
        -------
        - A `list` of `list` of `tensors` that represents the predicted
        displacement fields per pyramid, per each variational unit. - Same as
        previous, but represents the predicted flow fields. - Same as previous,
        but represents the registered batch of images.
        """

        is_legit_shape(
            ShapeBreakDown(fixed_images).core_shape_list,
            self.config.pyramid_levels_count,
            self.config.displacements_control_points_spacings,
            True,
        )

        # Create list containing multiple images, if pyramid is activated
        fixed_images_levels: List[tf.Tensor] = self._pyramid_images_generator(
            fixed_images, self.config.pyramid_levels_count
        )

        moving_images_levels: List[tf.Tensor] = self._pyramid_images_generator(
            moving_images, self.config.pyramid_levels_count
        )

        pipeline_data = PipeLineData()

        # Iterate over all the levels of the Pyramid
        # from the smallest/coarsest level ...
        for level_index, pyramid_level in enumerate(self._pyramid_levels):
            initial_vu_data = self._get_initial_vu(
                fixed_images_levels,
                moving_images_levels,
                level_index,
                pipeline_data.last_pyramid(),
            )

            pl_data = pyramid_level.push(initial_vu_data)

            pipeline_data.pyramids.append(pl_data)

        return pipeline_data

    def backward_propagate(self, return_gradients : bool, return_variables : bool) -> Tuple[Optional[List[NDArray]], Optional[List[NDArray]]]:
        """
        Back propagates through the network, and updates the model gradients,
        and makes it learn.

        Arguments
        ---------
        - `return_gradients`: If `True`, returns a `list` of `arrays` of the
        model gradients.
        - `return_variables`: If `True`, returns a `list` of `arrays` of the
        updated model parameters after back propagation.

        Returns
        -------
        - A `list` of `arrays` of the model gradients.
        - A `list` of `arrays` of the updated model parameters after back
          propagation.

        Note
        ----
        - Should be called only after calling `forward_propagate` function.
        """
        # wrong custom model setup can lead to NaN values in gradients

        for variables in self.vars.to_list():
            assert not tf.reduce_any(
                tf.math.is_nan(variables)
            ).numpy(), "Tensor contains NaN values"

        gradients = self._tensorflow_tape.gradient(self._loss, self.vars.to_list())

        grads : Optional[List[NDArray]] = None
        if return_gradients:
            # they will be disposed after apply_gradients
            grads = [g.numpy() for g in gradients]

        for variables in self.vars.to_list():
            assert not tf.reduce_any(
                tf.math.is_nan(variables)
            ).numpy(), "Tensor contains NaN values"

        for grad in gradients:
            assert not tf.reduce_any(
                tf.math.is_nan(grad)
            ).numpy(), "Tensor contains NaN values"

        self.config.optimizer.apply_gradients(zip(gradients, self.vars.to_list()))

        for variables in self.vars.to_list():
            assert not tf.reduce_any(
                tf.math.is_nan(variables)
            ).numpy(), "Tensor contains NaN values"

        return grads, self.vars.to_numpy() if return_variables else None

    def forward_propagate(
        self,
        fixed_images: tf.Tensor,
        moving_images: tf.Tensor,
        is_learning: bool = False,
        extra_loss_function_args: Optional[List[Any]] = None,
    ) -> Tuple[
        PipeLineData,
        tf.Tensor,
        List[Any],
    ]:
        """
        Performs forward propagation of batch.

        Arguments
        ---------
        - `fixed_images`: a batch of fixed images.
        - `moving_images`: a batch of moving images.
        - `is_learning`: If true, the model will track changes to calculate
        gradients during backward propagation.
        - `extra_loss_function_args`: extra arguments that will be passed
        to the surrogate loss function.

        Returns
        -------
        - A `PipeLineData` object that holds input/output data of whole pipeline.
        - The loss of the current batch as reported by the surrogate loss function.
        - A list of any extra variables reported by the surrogate loss function.
        """
        assert tf.reduce_any(tf.shape(fixed_images) - tf.shape(moving_images) == 0)

        # if we are not in learning stage, there is no need to register a tape.
        # note that tape should be recreated every time, since tensorflow
        # discards it once the gradients are applied in backward_propagate.

        # to create a conditional with context for tape see
        # https://stackoverflow.com/questions/27803059/conditional-with-statement-in-python
        if is_learning:
            self._tensorflow_tape = tf.GradientTape(watch_accessed_variables=False)
        else:
            self._tensorflow_tape = nullcontext()

        # if is_learning:
        with self._tensorflow_tape:
            if is_learning:
                self._tensorflow_tape.watch(self.vars.to_list())

            pipeline_data = self._forward_propagate(fixed_images, moving_images)

            self._loss, extra = self.config.surrogate_loss_function_instance.loss(
                pipeline_data,
                extra_loss_function_args,
            )

        return pipeline_data, self._loss, extra

    def _get_initial_vu(
        self,
        fixed_images_pyramid: List[tf.Tensor],
        moving_images_pyramid: List[tf.Tensor],
        pyramid_level: int,
        previous_pl_data: Optional[PyramidLevelData],
    ) -> VariationalUnitData:
        """
        Creates an initial/virtual VU of next PL.

        Notes
        -----
        - This simulates a new PL input to the model, it will not be saved to `PipeLineData` object.
        """

        initial_vu_data = VariationalUnitData()

        initial_vu_data.fixed_images = fixed_images_pyramid[pyramid_level]
        initial_vu_data.moving_images = moving_images_pyramid[pyramid_level]

        if previous_pl_data is not None:
            base_vu_data = previous_pl_data.last_variation_unit().clone()

            # clone displacements and upgrade them
            initial_vu_data.predicted_displacements = self._upgrader(
                base_vu_data.predicted_displacements
            )

        else:
            shape = ShapeBreakDown(initial_vu_data.fixed_images)

            # calc initial displacements shape
            displacements_core_shape = tf.cast(
                shape.core_shape, self.config.float_data_type
            )
            displacements_core_shape = (
                displacements_core_shape
                // self.config.displacements_control_points_spacings_tf
            )
            displacements_core_shape = displacements_core_shape.numpy().astype(np.int32)

            initial_vu_data.predicted_displacements = tf.zeros(
                [shape.batch_size_int]
                + list(displacements_core_shape)
                + [shape.core_rank_int],
                dtype=self.config.float_data_type,
            )

        return initial_vu_data
