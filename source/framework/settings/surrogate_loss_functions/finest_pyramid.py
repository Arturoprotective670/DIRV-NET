from typing import List, Tuple
import tensorflow as tf

from source.framework.tools.general import get_dissimilarity

class FinestPyramidLoss():
    def __init__(self, config) -> None:
        self._config = config

        self._fixed_images = None
        self._moving_images = None

        self._biggest_pyramid_per_unit_predicted_inverse_transformations = None
        self._biggest_pyramid_per_unit_registered_images = None

        self._ground_truth_inverse_transformations = None

        self._intensity_dissimilarity_loss = 0
        self._flow_field_dissimilarity_loss = 0

    def loss(self,
            fixed_images,
            ground_truth_inverse_transformations,
            per_pyramid_registered_images, \
            per_pyramid_predicted_inverse_transformations, \
            args : List = None):
        """
        Calculates the losses of the forward propagated images.

        Arguments
        ---------
        - `ground_truth_inverse_flow_fields`: a Tensor of a batch of the inverse
        ground truth of the flow fields.
        - `temperature`: If provided, the network is trained with exponential
        layer decay
        https://arxiv.org/abs/1906.05528.
        Describes the temperature parameter for exponential weighting.

        """
        self._biggest_pyramid_per_unit_predicted_inverse_transformations = \
                            per_pyramid_predicted_inverse_transformations[-1]
        self._ground_truth_inverse_transformations = \
                            ground_truth_inverse_transformations
        self._biggest_pyramid_per_unit_registered_images = \
                            per_pyramid_registered_images[-1]
        self._fixed_images = fixed_images

        # units count could be different from config if we have parameter sharing
        units_count = len(self._biggest_pyramid_per_unit_registered_images)

        intensity_dissimilarity_loss = 0.0
        inverse_transformations_dissimilarity_loss = 0.0
        temperature = args[0] if args is not None else None

        if temperature is not None:

            for unit_index in tf.range(units_count):
                unit_index = tf.cast(unit_index, tf.float32)

                exponential_loss_multiplier = tf.exp(-(units_count - \
                                            1.0 - unit_index) * temperature)

                losses = self._calc_unit_losses(unit_index)

                intensity_dissimilarity_loss += \
                                        losses[1] * exponential_loss_multiplier
                inverse_transformations_dissimilarity_loss += \
                                        losses[0] * exponential_loss_multiplier

        else:
            inverse_transformations_dissimilarity_loss, \
            intensity_dissimilarity_loss = \
                                        self._calc_unit_losses(units_count - 1)

        combined_losses = \
            self._config.transformations_dissimilarity_loss_weight * \
                inverse_transformations_dissimilarity_loss + \
                    self._config.intensity_dissimilarity_loss_weight * \
                        intensity_dissimilarity_loss

        return tf.reduce_mean(combined_losses), \
            [inverse_transformations_dissimilarity_loss, \
                intensity_dissimilarity_loss]

    def _calc_unit_losses(self, unit_index : int):
        """
        A help function, that calculates the losses within a specific network
        layer.

        Arguments
        ---------
        - `unit_index`: The index of the variational unit that we need to calculate.

        Returns
        -------
        - Mean image intensity dissimilarity loss across the batch.
        - Mean flow field intensity dissimilarity across the batch.
        """
        predicted_inverse_transformations = self. \
                _biggest_pyramid_per_unit_predicted_inverse_transformations[unit_index]

        # Since the number of pixels in all images of single batch, are the same
        # by definition, taking the mean of all batch pixels, or taking the mean
        # of each image, then the mean of the batch, will give the same result

        # ground_truth_flow_fields could be not provided
        if self._ground_truth_inverse_transformations is None:
            inverse_transformations_dissimilarity_loss = 0
        else:
            # calc mean per image
            inverse_transformations_dissimilarity_loss = \
                get_dissimilarity( \
                    predicted_inverse_transformations ,\
                            self._ground_truth_inverse_transformations)

        registered_images = self._biggest_pyramid_per_unit_registered_images[unit_index]

        intensity_dissimilarity_loss = \
                get_dissimilarity(self._fixed_images, registered_images)

        return inverse_transformations_dissimilarity_loss, \
                    intensity_dissimilarity_loss

    def batch_initial_losses(self, batch) -> \
                                        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Calculates the initial loses of the given batch
        """
        initial_intensity_losses = \
            get_dissimilarity(batch.fixed_images, batch.moving_images)

        initial_transformation_losses = \
            get_dissimilarity(batch.ground_truth_inverse_transformations, \
                        tf.zeros_like(batch.ground_truth_inverse_transformations))

        initial_losses = \
            self._config.transformations_dissimilarity_loss_weight * \
                initial_transformation_losses + \
                    self._config.intensity_dissimilarity_loss_weight * \
                        initial_intensity_losses

        initial_losses = tf.reduce_mean(initial_losses)

        return initial_losses, \
            [initial_transformation_losses, \
                initial_intensity_losses]
