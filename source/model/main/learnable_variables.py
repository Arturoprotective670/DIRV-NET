from typing import Any, List, Optional, Union

from numpy.typing import NDArray

import tensorflow as tf

from source.model.settings.enums import OptimizationApproaches

from source.model.settings.config import Config


class LearnableVariables:
    """
    Purpose
    -------
    - Initializes and hosts the DIRV-Net learnable parameters.

    Contributors
    ------------
    - TMS-Namespace
    - Claudio Fanconi
    """

    def __init__(self) -> None:
        self.data_term_kernels: Union[NDArray, tf.Tensor] = None
        self.data_term_potential_function_parameters: Union[NDArray, tf.Tensor] = None

        self.regularization_term_kernels: Union[NDArray, tf.Tensor] = None
        self.regularization_term_potential_function_parameters: Union[
            NDArray, tf.Tensor
        ] = None

        self.learning_rates: Union[NDArray, tf.Tensor] = None

        self.momentum_optimization_betas: Optional[Union[NDArray, tf.Tensor]] = None

        self._list: Optional[List[Union[NDArray, tf.Variable]]] = None

    def to_numpy(self) -> "LearnableVariables":
        result = LearnableVariables()

        result.data_term_kernels = self.data_term_kernels.numpy()
        result.data_term_potential_function_parameters = (
            self.data_term_potential_function_parameters.numpy()
        )

        result.regularization_term_kernels = self.regularization_term_kernels.numpy()
        result.regularization_term_potential_function_parameters = (
            self.regularization_term_potential_function_parameters.numpy()
        )

        result.learning_rates = self.learning_rates.numpy()

        if self.momentum_optimization_betas is not None:
            result.momentum_optimization_betas = (
                self.momentum_optimization_betas.numpy()
            )

        return result

    def to_list(self) -> List[Union[NDArray, tf.Variable]]:
        if self._list is None:
            self._list = []

            self._list.append(self.data_term_kernels)
            self._list.append(self.data_term_potential_function_parameters)

            self._list.append(self.regularization_term_kernels)
            self._list.append(self.regularization_term_potential_function_parameters)

            self._list.append(self.learning_rates)

            if self.momentum_optimization_betas is not None:
                self._list.append(self.momentum_optimization_betas)

        return self._list

    def initialize_variables(self, config: Config) -> None:
        tf.random.set_seed(config.randomizing_seed)

        cross_unit_parameter_sharing_count_tf = tf.constant(
            config.cross_units_parameter_sharing_count,
            dtype=config.int_data_type,
        )

        apparent_unit_count = (
            config.unfoldings_count / cross_unit_parameter_sharing_count_tf
        )

        images_rank_tensor = tf.constant(config.images_rank, dtype=config.int_data_type)
        # this is a repeated part of the shapes usd below
        main_shape = [config.pyramid_levels_count, int(apparent_unit_count)]

        # === data term kernels
        std = (images_rank_tensor / config.convolutional_kernels_count) ** (
            1 / images_rank_tensor
        ) / config.data_term_convolution_kernel_size
        shape = [
            config.channels_count,
            *([config.data_term_convolution_kernel_size] * config.images_rank),
            config.convolutional_kernels_count,
        ]

        self._to_tensorflow_variable(
            tf.random.normal(
                main_shape + shape,
                stddev=tf.cast(std, dtype=config.float_data_type),
                dtype=config.float_data_type,
            ),
            "data_term_kernels",
            config,
        )

        # === data potential functions
        shape = [
            config.channels_count,
            config.potentials_parameters_count,
            config.convolutional_kernels_count,
        ]
        self._to_tensorflow_variable(
            tf.random.truncated_normal(
                main_shape + shape,
                stddev=config.potentials_random_initialization_standard_deviation,
                dtype=config.float_data_type,
            ),
            "data_term_potential_function_parameters",
            config,
        )

        # === Regularizer filter
        std = (images_rank_tensor / config.convolutional_kernels_count) ** (
            1 / images_rank_tensor
        ) / config.regularization_term_convolution_kernel_size
        shape = [
            config.images_rank,
            *(
                [config.regularization_term_convolution_kernel_size]
                * config.images_rank
            ),
            config.convolutional_kernels_count,
        ]
        self._to_tensorflow_variable(
            tf.random.normal(
                main_shape + shape,
                stddev=tf.cast(std, dtype=config.float_data_type),
                dtype=config.float_data_type,
            ),
            "regularization_term_kernels",
            config,
        )

        # === regularizer potential functions
        shape = [
            config.images_rank,
            config.potentials_parameters_count,
            config.convolutional_kernels_count,
        ]
        self._to_tensorflow_variable(
            tf.random.truncated_normal(
                main_shape + shape,
                stddev=config.potentials_random_initialization_standard_deviation,
                dtype=config.float_data_type,
            ),
            "regularization_term_potential_function_parameters",
            config,
        )

        # === learning rates
        self._to_tensorflow_variable(
            tf.random.uniform(
                main_shape,
                minval=config.learning_rates_random_initialization_range[0],
                maxval=config.learning_rates_random_initialization_range[1],
                dtype=config.float_data_type,
            ),
            "learning_rates",
            config,
        )

        # === Momentum Rates
        if config.optimization_approach == OptimizationApproaches.POLYAK_HEAVY_BALL:
            self._to_tensorflow_variable(
                tf.random.uniform(
                    main_shape,
                    minval=config.learning_rates_random_initialization_range[0],
                    maxval=config.learning_rates_random_initialization_range[1],
                    dtype=config.float_data_type,
                ),
                "momentum_optimization_betas",
                config,
            )

    def _to_tensorflow_variable(
        self, parameters: tf.Tensor, variable_name: str, config: Config
    ) -> None:
        """
        Converts tensorflow tensor to tensorflow variable, and adds it to model
        watchable and learnable variables' collection
        """
        tensorflow_var = tf.Variable(
            parameters, dtype=config.float_data_type, name=variable_name
        )

        # self.vars_list.append(tensorflow_var)
        setattr(self, variable_name, tensorflow_var)
