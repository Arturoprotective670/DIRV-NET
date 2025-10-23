from typing import Any, List, Optional, Tuple
import tensorflow as tf

from source.model.main.pipe_line.pipe_line_data import (
    PipeLineData,
    PyramidLevelData,
    VariationalUnitData,
)
from source.model.tools.operators.resizing_operator import ResizingOperator
from source.model.tools.operators.upgrading_operator import UpgradingOperator

from source.framework.main.data_management.batch import Batch
from source.framework.main.data_management.losses_storage import LossesStorage
from source.framework.tools.general import get_dissimilarity
from source.framework.settings.whole_config import WholeConfig

from source.model.settings.surrogate_loss_interface import SurrogateLossInterface


class PerPyramidsLastVULoss(SurrogateLossInterface):
    """
    Purpose
    -------
    Surrogate loss function, per pyramid level, for last variational unit. (see eq. 22)

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(self, config: WholeConfig) -> None:
        super().__init__()

        self._config: WholeConfig = config

        self._displacements_pyramid_generator = UpgradingOperator(
            config.images_rank,
            config.displacements_upgrading_down_method,
            config.displacements_upgrading_up_method,
        ).create_pyramid

        self._images_pyramid_generator = ResizingOperator(
            config.images_rank, config.images_resizing_method
        ).create_pyramid

    def loss(
        self,
        pipeline_data: PipeLineData,
        args: Optional[List[Any]] = None,
    ) -> Tuple[tf.Tensor, List[Any]]:
        """
        Calculates the losses of the forward propagated images.

        Arguments
        ---------
        - `pipeline_data`: The pipeline data objects, that holds information of
          all pipeline stages.
        - `args`: a list of variables that we what to pass to loss function, in
          this loss function, they will contain two variables:
            - `ground_truth_displacements`: a Tensor of a batch of the ground
              truth displacements fields.
            - `temperature`: If provided, the network is trained with
              exponential layer decay

        Returns
        -------
        - Scalar tensor with the loss value.
        - `LossesStorage` object.
        """
        # https://arxiv.org/abs/1906.05528.
        # Describes the temperature parameter for exponential weighting.

        ground_truth_displacements = args[0]

        # if GT fields are too small, consider them absent
        if (
            tf.reduce_max(tf.math.abs(ground_truth_displacements))
            <= self._config.synthetic_flow_fields
        ):
            ground_truth_displacements = tf.zeros_like(ground_truth_displacements)

        ground_truth_displacements_levels: List[tf.Tensor] = self._get_pyramid(
            ground_truth_displacements, True
        )

        per_level_per_image_displacements_loss: List[tf.Tensor] = []
        per_level_per_image_intensity_loss: List[tf.Tensor] = []

        for pl_index, pl_data in enumerate(pipeline_data.pyramids):
            # populate last variation units GT displacements
            for vu_data in pl_data.variational_units:
                vu_data.ground_truth_displacements = ground_truth_displacements_levels[
                    pl_index
                ]

            level_per_image_displacements_dissimilarity_loss: tf.Tensor = (
                get_dissimilarity(
                    pl_data.last_variation_unit().ground_truth_displacements,
                    pl_data.last_variation_unit().predicted_displacements,
                )
            )

            level_per_image_intensity_dissimilarity_loss: tf.Tensor = get_dissimilarity(
                pl_data.last_variation_unit().fixed_images,
                pl_data.last_variation_unit().registered_images,
            )

            # unify dissimilarity loss scale, that happens due to the upgrading
            dissimilarity_loss_scaling_factor: float = (
                self._config.pyramid_levels_up_scaling
                ** (self._config.pyramid_levels_count - 1 - pl_index)
            )
            level_per_image_displacements_dissimilarity_loss *= (
                dissimilarity_loss_scaling_factor
            )

            # we want intensity loss to be in range [0, 255] instead of [0, 1],
            # as usual image pixel values.
            level_per_image_intensity_dissimilarity_loss *= 255

            per_level_per_image_displacements_loss.append(
                level_per_image_displacements_dissimilarity_loss
            )
            per_level_per_image_intensity_loss.append(
                level_per_image_intensity_dissimilarity_loss
            )

        # per_image_combined_loss = self._per_image_combined_losses(
        #     tf.reduce_mean(per_level_per_image_displacements_loss, axis=0),
        #     tf.reduce_mean(per_level_per_image_intensity_loss, axis=0),
        # )

        per_image_combined_loss = tf.reduce_mean(
            per_level_per_image_displacements_loss, axis=0
        )

        assert not tf.reduce_any(
            tf.math.is_nan(per_image_combined_loss)
        ).numpy(), "Tensor contains NaN values"

        storage = LossesStorage()
        storage.merge_losses(
            per_image_combined_loss,
            per_level_per_image_displacements_loss,
            per_level_per_image_intensity_loss,
        )

        return tf.reduce_mean(per_image_combined_loss), [storage]

    def _get_pyramid(
        self, source: tf.Tensor, is_displacements: bool
    ) -> List[tf.Tensor]:
        if is_displacements:
            pyramids = self._displacements_pyramid_generator(
                source, self._config.pyramid_levels_count
            )
        else:
            pyramids = self._images_pyramid_generator(
                source, self._config.pyramid_levels_count
            )

        return pyramids

    def _per_image_combined_losses(
        self,
        displacements_dissimilarity_loss: tf.Tensor,
        intensity_dissimilarity_loss: tf.Tensor,
    ) -> tf.Tensor:
        # revert intensities to [0,1]
        # intensity_dissimilarity_loss /= 255

        # # intensity dis. loss can have values in range [0,1], but dissimilarity
        # # loss can has be any, so if we want use both, and to make weight
        # # parameter something reasonable, we should scale up intensity dis.
        # # to displacements magnitude.

        # intensity_actual_weight = self._config.intensity_dissimilarity_loss_weight

        # if self._config.intensity_dissimilarity_loss_weight != 0:
        #     scale_factor = tf.reduce_max(tf.abs(displacements_dissimilarity_loss))
        #     intensity_actual_weight *= scale_factor

        return intensity_dissimilarity_loss
        # intensity_actual_weight = 0.3
        # return (
        #     (1 - self._config.intensity_dissimilarity_loss_weight)
        #     * displacements_dissimilarity_loss
        #     + intensity_actual_weight * intensity_dissimilarity_loss
        # )

    def batch_initial_losses(self, batch: Batch) -> LossesStorage:
        """
        Calculates the initial loses of the given batch.
        """

        registered_images_pyramid = self._get_pyramid(batch.moving_images, False)

        displacements_pyramid = self._get_pyramid(
            tf.zeros_like(batch.ground_truth_displacements), True
        )

        fixed_images_pyramid = self._get_pyramid(batch.fixed_images, False)

        pipeline_data = PipeLineData()

        for i in range(self._config.pyramid_levels_count):
            pl_data = PyramidLevelData()

            for _ in range(self._config.unfoldings_count):
                vu_data = VariationalUnitData()

                vu_data.fixed_images = fixed_images_pyramid[i]
                vu_data.registered_images = registered_images_pyramid[i]
                vu_data.predicted_displacements = displacements_pyramid[i]

                pl_data.variational_units.append(vu_data)

            pipeline_data.pyramids.append(pl_data)

        _, loss_storage = self.loss(
            pipeline_data,
            [batch.ground_truth_displacements],
        )

        return loss_storage[0]
