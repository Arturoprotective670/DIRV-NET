import tensorflow as tf

from source.model.main.pipe_line.pipe_line_data import VariationalUnitData
from source.model.main.learnable_variables import LearnableVariables
from source.model.settings.enums import OptimizationApproaches

from source.model.main.pipe_line.field_of_experts.foe_data_term import FoEDataTerm
from source.model.main.pipe_line.field_of_experts.foe_regularization_term import (
    FoERegularizationTerm,
)

from source.model.settings.config import Config

from source.model.tools.linear_smoothing_kernel import (
    first_degree_b_spline_derivative_filter,
)
from source.model.tools.operators.warping_operator import warp_images

from source.model.tools.operators.image_gradient_operator import images_gradients
from source.model.tools.operators.resizing_operator import ResizingOperator
from source.model.tools.operators.sampling_operator import sample


class VariationalUnit:
    """
    Purpose
    --------
    - Represents a variational unit object in the DIRV-Net (see eq. 19, 25, 28).

    Contributors
    ------------
    - TMS-Namespace
    - Claudio Fanconi
    """

    def __init__(
        self,
        config: Config,
        learnable_variables: LearnableVariables,
        pyramid_level: int,
        vu_index: int,
    ) -> None:
        self._config = config

        self._py_index = pyramid_level
        self._vu_index = vu_index

        self._displacements_make_like = ResizingOperator(
            config.images_rank, config.displacements_to_flows_resizing_up_method
        ).make_like

        # Smoothing kernel for Adjoint BSpline Operation
        self._spline_derivative_filter = first_degree_b_spline_derivative_filter(
            self._config.displacements_control_points_spacings, config.float_data_type
        )

        self._vars = learnable_variables

        self._FoE_data = FoEDataTerm(
            config, learnable_variables, pyramid_level, vu_index
        ).Push
        self._FoE_regularization = FoERegularizationTerm(
            config, learnable_variables, pyramid_level, vu_index
        ).Push

    def _convolve_image_level_force(self, image_level_force: tf.Tensor) -> tf.Tensor:
        """
        Convolve every image level force component, with control points force
        (aka. the Spline linear smoothing kernel) (see eq. 29).
        """

        strides = [1] * (self._config.images_rank + 2)
        padding = "SAME"

        res = []

        for axis in range(self._config.images_rank):
            force = tf.expand_dims(image_level_force[..., axis], axis=-1)
            conv = self._config.convolve.convolve(
                force, self._spline_derivative_filter, strides, padding
            )
            res.append(conv[..., 0])

        return tf.stack(res, axis=-1) # our smoothed image force

    def _data_term_gradient(
        self,
        displacements: tf.Tensor,
        fixed_images: tf.Tensor,
        moving_images: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Approximated data term gradient of the VU (see eqs. 19, 28).

        Returns
        -------
        - VU data term gradient tensor.
        - Registered images of the previous VU.
        - Interpolated flow fields of the previous VU.
        """
        # === calc the flows and registered images
        input_flows = self._displacements_make_like(displacements, moving_images)

        input_registered_images, _ = warp_images(
            moving_images, input_flows, self._config.fields_min_threshold
        )

        foe = self._FoE_data(input_registered_images - fixed_images)

        # batch, core, rank
        foe = tf.tile(
            foe, [1] * (self._config.images_rank + 1) + [self._config.images_rank]
        )

        # === Warped Image gradient
        # 2x [batch, core, 1]
        gradients_list = images_gradients(input_registered_images)
        # batch, core, rank
        registered_images_nabla = tf.stack(gradients_list, axis=-1)[..., 0, :]

        # ===  Calc image level force
        # batch, core, rank
        image_level_force = foe * registered_images_nabla

        # batch, core, rank
        vu_data_gradient = self._convolve_image_level_force(image_level_force)

        # === sample only at the control points
        vu_data_gradient = sample(
            vu_data_gradient, self._config.displacements_control_points_spacings
        )

        return vu_data_gradient, input_registered_images, input_flows

    def push(self, previous_vu_data: VariationalUnitData) -> VariationalUnitData:
        """
        Pushes data through the variational unit pipeline (see eq. 25).
        """

        (
            data_term_gradient,
            previous_vu_data.registered_images,
            previous_vu_data.predicted_flows,
        ) = self._data_term_gradient(
            previous_vu_data.predicted_displacements,
            previous_vu_data.fixed_images,
            previous_vu_data.moving_images,
        )

        regularization_term_gradient = self._FoE_regularization(
            previous_vu_data.predicted_displacements
        )

        loss_function_gradient = data_term_gradient + regularization_term_gradient

        learning_rate = tf.abs(
            self._vars.learning_rates[self._py_index, self._vu_index, ...]
        )

        predicted_displacements = previous_vu_data.predicted_displacements

        if (
            self._config.optimization_approach
            == OptimizationApproaches.GRADIENT_DESCENT
        ):
            predicted_displacements -= learning_rate * loss_function_gradient

        elif (
            self._config.optimization_approach
            == OptimizationApproaches.POLYAK_HEAVY_BALL
        ):  # TODO:
            beta = tf.abs(self._config.momentum_optimization_betas)
            diff = (
                previous_vu_data.predicted_displacements
                - previous_vu_data.input_displacements
            )
            predicted_displacements -= (
                learning_rate * loss_function_gradient + beta * diff
            )

        vu_data = VariationalUnitData()
        vu_data.fixed_images = previous_vu_data.fixed_images
        vu_data.moving_images = previous_vu_data.moving_images
        vu_data.input_displacements = previous_vu_data.predicted_displacements
        vu_data.input_flows = previous_vu_data.predicted_flows
        vu_data.predicted_displacements = predicted_displacements

        return vu_data
