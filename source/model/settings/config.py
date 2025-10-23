import tensorflow as tf
from typing import Literal, Tuple, Callable, Union

from source.model.settings.enums import (
    OptimizationApproaches,
    PotentialsParametrizationModes,
)

from source.model.tools.operators.resizing_operator import ResizingMethods
from source.model.tools.operators.convolution_operator import ConvolutionOperator

from source.model.settings.surrogate_loss_interface import SurrogateLossInterface


class Config:
    """
    Purpose
    -------
    - The core configuration class of DIRV-Net.
    - Initializes the used optimizer.
    - Initializes the suitable conversion operators.

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(self) -> None:
        # the number of filters/kernels that will be used for both data and
        # regularization terms.
        # TODO: Split them.
        self.convolutional_kernels_count: int = 16
        # Knots count that will be used to parametrize the activation/potential
        # functions, in a line-wise way.
        self.potentials_parameters_count: int = 24
        # to reduce the number of parameters in the network, we can share the
        # parameters among consequent variational units, set to 1 to disable.
        self.cross_units_parameter_sharing_count: int = 0

        # convolutional kernels sizes
        self.data_term_convolution_kernel_size: int = 7
        self.regularization_term_convolution_kernel_size: int = 5

        # unfoldings count, or the number of variational units per pyramid level.
        self.unfoldings_count: int = 5
        # the core image rank that the network should expect, needed for
        # initialization.
        self.images_rank: int = 2
        # images channels count, currently ony mono-channel images are supported.
        self.channels_count: int = 1
        # the number of pyramid levels, that should be generated for the network.
        self.pyramid_levels_count: int = 3

        # Surrogate loss function, that should accept and return the following
        # variables:
        self.surrogate_loss_function_instance: SurrogateLossInterface = None

        # how sparse the displacement fields should be, the smaller the more fine
        # displacements the network will learn, and more resources it will need.
        self.displacements_control_points_spacings: Union[
            Tuple[int, int], Tuple[int, int, int]
        ] = (4, 4)

        # below is for random model parameters initialization.
        self.randomizing_seed: int = 0
        self.learning_rates_random_initialization_range: Tuple[int, int] = (0, 0.1)
        self.potentials_random_initialization_standard_deviation: float = 0.01

        self.float_data_type: tf.DType = tf.float32
        self.int_data_type: tf.DType = tf.int32

        # the used optimization approach (i.e. how the learning rates (that are
        # actually model parameters, not hyperparameters) should be treated).
        self.optimization_approach: OptimizationApproaches = (
            OptimizationApproaches.GRADIENT_DESCENT
        )
        # how the potential functions will be interpolated.
        self.potentials_parametrization_mode: PotentialsParametrizationModes = (
            PotentialsParametrizationModes.PIECE_WISE_LINEAR
        )
        # on which region the potential functions should be considered
        self.potential_functions_definition_region: Tuple[float, float] = (0, 1)

        # the resizing method for image pyramid levels
        self.images_resizing_method: ResizingMethods = ResizingMethods.LINEAR_SMOOTHED
        # how upsampling the predicted displacements for next pyramid level
        self.displacements_upgrading_up_method: ResizingMethods = (
            ResizingMethods.NEAREST_NEIGHBOR
        )
        # how GT displacements pyramid generated
        self.displacements_upgrading_down_method: ResizingMethods = (
            ResizingMethods.SMOOTHED_NEAREST_NEIGHBOR
        )
        # how i.e. warping is applied
        self.displacements_to_flows_resizing_up_method: ResizingMethods = (
            ResizingMethods.B_SPLINES
        )

        # we use only adam optimizer, below is it's parameters, and the
        # optimizer is initialized in _init() function.
        self.optimizer: Callable = None
        self.adam_optimizer_learning_rate: float = 2e-4
        self.adam_optimizer_beta1: float = 0.9
        self.adam_optimizer_beta2: float = 0.999

        # do not warp if field components value/strength is less than (see warping operator)
        self.fields_min_threshold: float = 0.002

        # internal model variables
        self.displacements_control_points_spacings_tf: tf.Tensor = None
        self._cross_unit_parameter_sharing_count_tf: tf.Tensor = None

        # currently no other factors are supported
        self._pyramid_level_scaling_factor: int = 2

        self.convolve: ConvolutionOperator = None

    def _init(self) -> None:
        # === Validate Parameters Logical Consistency
        if self.channels_count != 1:
            raise Exception("Only mono-channel images are supported currently.")

        if self.pyramid_levels_count < 1:
            raise Exception("Pyramids count should be at least one.")

        if any(item < 1 for item in self.displacements_control_points_spacings):
            raise Exception(
                "Incorrect spacing between control points of displacements."
            )

        if self.unfoldings_count % self.cross_units_parameter_sharing_count != 0:
            raise Exception(
                "Layers that share parameters is not a multiple"
                + "of total layers count."
            )

        # TODO: add more validation rules

        # === setup internal vars
        self.displacements_control_points_spacings_tf = tf.constant(
            self.displacements_control_points_spacings, dtype=self.float_data_type
        )

        self._cross_unit_parameter_sharing_count_tf = tf.constant(
            self.cross_units_parameter_sharing_count, dtype=self.int_data_type
        )

        self.convolve = ConvolutionOperator(self.images_rank)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.adam_optimizer_learning_rate,
            beta_1=self.adam_optimizer_beta1,
            beta_2=self.adam_optimizer_beta2,
        )
