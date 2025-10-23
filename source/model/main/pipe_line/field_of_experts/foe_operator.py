import tensorflow as tf

from source.model.settings.config import Config
from source.model.settings.enums import PotentialsParametrizationModes

from source.model.tools.potential_functions import (
    potential_function_RBF,
    potential_function_piece_wise_linear,
)


class FieldsOfExpertsOperator:
    """
    Purpose
    --------
    - Field of Experts operator of DIRV-Net (see eq. 22).

    Contributors
    ------------
    - TMS-Namespace
    - Claudio Fanconi
    """

    def __init__(self, config: Config) -> None:
        self._config = config

        if (
            config.potentials_parametrization_mode
            == PotentialsParametrizationModes.PIECE_WISE_LINEAR
        ):
            self._apply_potential_function = potential_function_piece_wise_linear
        elif (
            config.potentials_parametrization_mode == PotentialsParametrizationModes.RBF
        ):
            self._apply_potential_function = potential_function_RBF
        else:
            raise NotImplementedError

    def Push(
        self, data: tf.Tensor, kernels: tf.Tensor, potentials: tf.Tensor
    ) -> tf.Tensor:
        """
        Calculates Field of Experts (FoE) output of the given data (see eq. 22).

        Arguments
        ---------
        - `data`: a `Tensor` of the format `[B, core_size, C]`, of
        the the date to calc FoE.
        - `kernels`: a `Tensor` of the format `[C, kernels_count, 1,
        filters_count]` of the set of kernels that will be used for convolution.
        - `potentials`: a `Tensor` of the format `[C, parameters_count,
        filters_count]` of the parameters that used to parametrize potential
        functions.

        Returns
        -------
        A `Tensor` of the format `[B, core_size, core_rank]`, that represents
        part of the estimated displacement field.

        Note
        ----
        Convolutions are performed on a per-channel basis.
        """
        per_channel_foe = []

        strides = [1] * (self._config.images_rank + 2)
        padding = "SAME"  # we need to keep core dimensions as of input

        # TODO: use ShapeBreakDown
        output_shape = tf.shape(data)
        channels_count = tf.shape(data)[-1]

        # we convolving only one channel, so we need to set this in output shape
        output_shape = list(output_shape.numpy())
        output_shape[-1] = 1

        # below is needed to keep input format as TF requires (i.e existence of
        # channels) during looping over them below
        data = tf.expand_dims(data, axis=-2)

        # the same should be done with filters, since convolution requires
        # input channels
        kernels = tf.expand_dims(kernels, axis=-2)

        # Convolve every channel separately
        for channel_index in tf.range(channels_count):
            kernel = kernels[channel_index, ...]

            # convolve data with kernel
            # outputs [B, core_size, filters_count]
            k_star_x = self._config.convolve.convolve(
                data[..., channel_index], kernel, strides, padding
            )

            # apply potential functions
            # outputs [B, core_size, filters_count]
            psi_k_conv_x = self._apply_potential_function(
                k_star_x,
                potentials[channel_index, ...],
                self._config.potential_functions_definition_region[0],
                self._config.potential_functions_definition_region[1],
                self._config.int_data_type,
            )

            # transpose convolve with the kernel
            # outputs [B, core_size, 1]
            channel_foe = self._config.convolve.convolve_transpose(
                psi_k_conv_x, kernel, output_shape, strides, padding
            )

            per_channel_foe.append(channel_foe)

        FoE = tf.stack(per_channel_foe, axis=-1)

        return FoE[..., 0, :]
