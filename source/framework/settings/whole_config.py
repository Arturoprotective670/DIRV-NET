from typing import List, Optional, Tuple, Union

from datetime import datetime

from source.model.settings.enums import (
    OptimizationApproaches,
    PotentialsParametrizationModes,
)
from source.model.settings.config import Config
from source.model.tools.operators.resizing_operator import ResizingMethods

from source.framework.tools.pather import Pather
from source.framework.tools.general import is_debug_mode

from source.framework.settings.enums import (
    DataSets,
    SyntheticFields,
    PreProcessingMethod,
    SurrogateLossFunction,
)


class WholeConfig(Config):
    """
    Purpose
    -------
    - A class that contains all the settings for the whole framework
    configuration, as well as for DIRV-Net.

    Notes
    -----
    - To avoid circular references issues, we import python files on the go in
      this class.

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        super().__init__()

        # region === Global settings

        self.force_GPU_ID: Optional[float] = None  # None for auto select
        self.max_used_gpu_memory: Optional[float] = None  # None to allocate all memory

        # Seed for reproducibility, used also for training
        self.global_random_seed: int = 42
        # Seed for testing reproducibility
        self.test_global_random_seed: int = 77

        # endregion

        # region === Session settings

        if session_id is None:
            # create a unique session ID, depending on startup time, and if in debug mode
            # this will used to how name the run-results folder
            self.session_id = (
                "debug_" if is_debug_mode() else "run_"
            ) + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.session_id = session_id

        # endregion

        # region === Output settings

        # current root session folder
        _session_pather = Pather("output", self.session_id)
        self.session_dir: str = _session_pather.directory

        # location of the config file, to be copied to output before session start
        self.config_file_path: str = Pather("source", "framework", "settings").join(
            "whole_config", "py"
        )

        # endregion

        # region === Data Set settings

        # Size of the training dataset
        self.crop_input_dataset_size_to: Optional[int] = 1_000
        self.crop_test_dataset_size_to: Optional[int] = 10_000

        self.shuffle_data_set: bool = False

        # split of training and test ratio, 0 for no validation
        self.train_to_validation_data_splitting_ratio: float = 0

        self.data_set: DataSets = DataSets.GRADIENT_CHESSBOARD_64

        # this is a trick to avoid the need of calculating the inverse fields
        # while generating the synthetic data that takes lots of calculation
        # resources when a good quality of the field inversion is needed (i.e.
        # when inverse_field_sampling_quality config is small), consequently,
        # this means that in this case no inverse fields are calculated.
        # However, this also means that the fields that network will learn to
        # predict, are simpler, since inverse of the fields are usually more
        # complex.
        self.swap_between_fixed_moving_images: bool = True

        # endregion

        # region === Surrogate Loss Function

        # weight of pixel intensity SSD loss in predicted image, set to zero to
        # use only dissimilarity field loss only
        self.intensity_dissimilarity_loss_weight: float = 0.5

        # exponential weighting in training, for every layer
        self.use_exponential_decay_training = False
        # increase step of temperature parameter of every iteration
        self.temperature_increment_step = 0.0001
        # starting temperature
        self.initial_temperature = 0.0
        # maximum temperature
        self.max_temperature = 2

        self.surrogate_loss_function: SurrogateLossFunction = (
            SurrogateLossFunction.PER_PYRAMID_LAST_VARIATIONAL_UNIT
        )

        # endregion

        # region === DIRV-Net Parameters

        self.channels_count = 1  # don't change, more channels are not supported
        self.images_rank = 2

        # number of layers in the DIRV-Net
        self.unfoldings_count = 7
        # set to 1 to not use pyramids, this number includes the original image
        self.pyramid_levels_count = 1
        # the relative size of pyramid levels, do not change!
        self.pyramid_levels_up_scaling = 2

        # TODO: split number of kernels for Data and realization terms
        self.convolutional_kernels_count = 16  # number of filters in the convolution
        self.data_term_convolution_kernel_size = 7  # filter size for data filters
        # filter size for flow field filters
        self.regularization_term_convolution_kernel_size = 5

        # number of interpolation knots in the activation functions
        self.potentials_parameters_count = 24

        # number of consequent variational units that will share same parameters
        self.cross_units_parameter_sharing_count = 1

        # prediction quality
        self.displacements_control_points_spacings = [4] * self.images_rank

        self.potentials_parametrization_mode = (
            PotentialsParametrizationModes.PIECE_WISE_LINEAR
        )
        self.potential_functions_definition_region = (-1, 1)
        self.optimization_approach = OptimizationApproaches.GRADIENT_DESCENT

        # below is the resizing method for image pyramid levels and plot generating
        self.images_resizing_method = ResizingMethods.LINEAR_SMOOTHED
        # below is used before applying displacements on the images (i.e. warping)
        self.displacements_to_flows_resizing_up_method = ResizingMethods.B_SPLINES

        # below is used for GT displacements pyramid-level calculation, as well
        # as for loss, and drawing
        self.displacements_upgrading_down_method = (
            ResizingMethods.SMOOTHED_NEAREST_NEIGHBOR
        )
        # below is used to upsample the predicted displacements for next pyramid level
        self.displacements_upgrading_up_method = ResizingMethods.NEAREST_NEIGHBOR

        # network initialization
        self.learning_rates_random_initialization_range = [
            0,
            0.1,
        ]  # we want them to be positive
        self.potentials_random_initialization_standard_deviation = 0.01

        # Adam Optimizer Parameters:
        # Initial learning rate
        self.adam_optimizer_learning_rate = 2e-4
        # first momentum parameter
        self.adam_optimizer_beta1 = 0.9
        # second momentum parameter
        self.adam_optimizer_beta2 = 0.999

        self.randomizing_seed = self.global_random_seed

        # endregion

        # region === Image pre-processing

        self.resized_image_size: List[int] = [64] * self.images_rank

        self.image_pre_processing_method: PreProcessingMethod = (
            PreProcessingMethod.GLOBAL_UNITY_NORMALIZATION
        )

        # endregion

        # region === Training/Testing Procedure

        # to test results on validation data at epoch end
        self.validate_data: bool = False

        # training length settings
        self.training_epochs_count: int = 2  # number of training epochs

        self.training_batch_size: int = 100
        self.validation_default_batch_size: int = 200
        self.testing_default_batch_size: int = 200

        # training patches settings
        # will generate random patches of images, None for using original images
        self.training_patches_shape: Union[Tuple[int, int], Tuple[int, int, int]] = [
            128
        ] * self.images_rank
        # set 0 for no patch generating
        self.training_patches_count: int = 0
        # patches that has std of intensity smaller than this number, will be
        # dropped, this makes the number of batches unpredictable, set zero to
        # disable checking
        self.empty_patch_max_std_threshold: float = 0.0

        # Parametrized step size scheduler, will use learning rate reduction if true
        self.use_parametrized_step_size_reduction: bool = False
        # Reduce learning rate after this amount of epochs
        self.parametrized_step_size_reduction_starting_epoch: int = 10
        # Reduce the learning rate by how much
        self.parametrized_step_size_reduction_factor: float = 1 / 2

        self.cross_layer_parameter_sharing_count: int = 1  # 1 for not sharing

        self.max_recurrent_refinement_iterations: int = 15

        # endregion

        # region === Synthetic fields

        # we usually want last epoch' batch to be same, so that we can
        # compare network learning progress at different epochs and runs
        self.freeze_last_batch: bool = True
        # for below smaller values increasing quality, and cpu & ram usage.
        # default value in source library is 16, note that this is still used
        # even with swapping trick off, during generating tensorboard images.
        self.inverse_field_sampling_quality = 8

        # see non_rigid_random_fields.py for details
        self.non_rigid_smoothing_kernels: List[Tuple[int, float]] = [(9, 3.0), (5, 2.0)]
        self.non_rigid_initial_seed_range: Tuple[int, int] = (30, 35)
        self.non_rigid_max_magnitude_range: Tuple[float, float] = (0.1, 6)

        # see affine_random_fields.py for details
        degree_to_radian = 2 * 3.1415926 / 360
        self.affine_origin_shift_range: Tuple[int, int] = (-2, 2)
        self.affine_translations_range: Tuple[float, float] = (-2, 2)
        self.affine_rotations_range: Tuple[float, float] = (
            -1.5 * degree_to_radian,
            1.5 * degree_to_radian,
        )
        self.affine_shears_range: Tuple[float, float] = (-0.04, 0.04)
        # we do not want inversion, so keep them positive
        self.affine_scales_ranges: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (0.98, 1.04),
            (0.96, 1.02),
        )

        self.synthetic_flow_fields: SyntheticFields = SyntheticFields.COMBINED

        # due to the numerous params above, getting a close to zero flows has a
        # low probability, to make the network learn also how to identify zero
        # flows, we will generate zero fields with below probability.
        self.white_noise_flows_replacing_probability: float = 0.001

        # do not warp if field components value/strength is less than (see
        # warping operator), it will be also used to generate white noise
        # fields, and below this value, the GT fields in loss function will be
        # teated as exactly zero.
        # Experiments showed that 0.005 will make max.
        # warping error less than 1 in pixel intensity
        self.white_noise_flows_threshold: float = 0.005

        # during moving images generating, we will add white noise to the flows
        # fields with below probability, but the reported to loss function GT
        # fields, will remain, by this we will teach the network to cancel out
        # white noise.
        self.white_noise_flows_poison_probability: float = 0.01

        # endregion

        # region === Logging
        # how often the progress of batch loop will be reported
        self.report_batches_progress_every: int = 10
        # how often the logs should be written in terms of lines
        self.flash_text_logs_every_lines: int = 1

        self.log_path: str = _session_pather.join("log", "txt")
        self.losses_saving_path: str = _session_pather.deeper("losses").directory
        self.save_losses_to_csv: bool = True

        # endregion

        # region === Model Variables Saving

        self.model_variables_saving_path: Pather = _session_pather.deeper(
            "saved_models"
        )
        self.start_from_epoch: int = 0  # starting epoch
        # we will save model variables every x epochs
        self.save_model_variables_every_epochs: int = 5
        # for the last x epochs, save variables every epoch
        self.save_model_variables_for_last_epochs: int = 5
        # endregion

        # region === TensorBoard reporting

        self.tensorboard_logging: bool = True  # Will create a TensorBoard logging file

        # images logging settings
        # the max. number of images to render in tensorboard report
        self.tensorboard_max_logged_images: int = 8

        # None for color inversion, should be in range [0,1]
        self.image_preview_grid_intensity: Optional[float] = 0.6
        # set to 1 to disable grid drawing
        self.image_preview_drawn_grid_lines_count: int = 1
        self.image_preview_cell_size_inches: Tuple[float, float] = [2.6] * 2
        self.image_preview_title_font_size: float = 10

        self.histograms_logging_cell_size_inches: Tuple[float, float] = [3.6] * 2
        self.histograms_bins_count: int = 100
        self.histograms_image_alpha: float = 0.6

        self.vector_fields_logging_cell_size_inches: Tuple[float, float] = [3] * 2

        self.potentials_logging_cell_size_inches: Tuple[float, float] = (7, 3.2)

        self.plotting_training_average_losses_every_batches: int = 20

        self.images_preview_color_map: str = "gray"
        self.displacements_preview_color_map: str = "bwr"
        self.images_diff_preview_color_map: str = "bwr"
        self.vector_fields_images_preview_color_map: str = "winter"

        _tensorboard_pather: Pather = _session_pather.deeper("tensorboard_logs")
        self.tensorboard_dir: str = _tensorboard_pather.directory

        self.train_log_dir: str = _tensorboard_pather.deeper("train").directory
        self.validation_log_dir: str = _tensorboard_pather.deeper(
            "validation"
        ).directory
        self.test_log_dir: str = _tensorboard_pather.deeper("test").directory
        self.inferring_log_dir: str = _tensorboard_pather.deeper("inferring").directory

        self.show_histogram_of_smallest_gradients_percentage: float = 75

        # profiling
        # a range of batches indexes that will be profield.
        # TF docs suggests to keep the range length around 10, and to not start
        # from the first batch.
        # Use negative numbers to disable profiling.
        self.batches_range_to_profile: Tuple[int, int] = (-5, -15)
        self.profiling_dir: str = _tensorboard_pather.deeper("profiling").directory

        # statistics box on diff/delta of images and fields options
        self.show_stats_box: bool = True
        self.stats_box_font_size: int = 9
        self.stats_box_font_color: Tuple[int, int, int, int] = (255, 255, 255, 50)
        self.stats_box_background_color: Tuple[int, int, int, int] = (79, 44, 14, 180)

        # endregion
