from typing import Callable, List, Optional, Tuple, Union

import time
from contextlib import nullcontext
import numpy as np

import tensorflow as tf

from source.framework.main.data_management.batch import Batch
from source.framework.main.data_management.data_set import Stages, DataSet
from source.framework.main.data_management.data_set_provider import DataSetProvider

from source.framework.main.refinement.recurrent_refinement import (
    per_batch_recurrent_refinement,
    per_image_recurrent_refinement,
)
from source.framework.main.refinement.recurrent_refinement_data import (
    RecurrentRefinementData,
)
from source.framework.settings.synthetic_fields_provider import SyntheticFieldsProvider
from source.framework.settings.whole_config import WholeConfig
from source.framework.settings.enums import SurrogateLossFunction

from source.framework.tools.gpu_tools import config_gpu
from source.framework.tools.pather import Pather

from source.framework.main.reporting.log_reporter import LogReporter
from source.framework.main.reporting.tensorboard_reporter import TensorBoardReporter

from source.model.main.dirv_net import DIRVNet


class ImageRegistration:
    """
    Purpose
    -------
    The top level class that controls the learning and inferring procedures,
    this includes loading and preparing data, loggers, saving the model, etc..

    Contributors
    ------------
    - TMS-Namespace
    - Claudio Fanconi
    """

    def __init__(self, config: WholeConfig) -> None:
        """
        Initialize `ImageRegisterer` class.

        Arguments
        ---------
        - `config`: the configuration class object.
        """
        # tf.config.set_visible_devices([], 'GPU')
        # tf.data.experimental.enable_debug_mode()

        # policy = tf.keras.mixed_precision.Policy('mixed_float16')
        # tf.keras.mixed_precision.set_global_policy(policy)
        # tf.keras.backend.set_floatx('float16')

        self._config = config

        self._logger = LogReporter(config)

        # this should be configured at the very beginning
        config_gpu(config.force_GPU_ID, config.max_used_gpu_memory)

        self._logger.report_gpu()

        # tf.config.run_functions_eagerly(not config.enable_auto_graphs)

        self._logger.log_line("Session ID: " + config.session_id)

        # to see job execution device (CPU/GPu) uncomment below
        # tf.debugging.set_log_device_placement(True)

        self._logger.log_line("Tensorflow version : " + tf.__version__)

        # save a copy of model config file, into current session dir, for
        # reference
        import shutil

        shutil.copy(config.config_file_path, config.session_dir)

        self._data_provider: DataSetProvider = None
        self._model: DIRVNet = None

        self._board = TensorBoardReporter(config)

        self._data_provider = DataSetProvider(self._config)

    # region === Help Functions

    def reset(self) -> None:
        self._logger = LogReporter(self._config)

        self._data_provider: DataSetProvider = None
        self._model: DIRVNet = None

        self._board = TensorBoardReporter(self._config)

        self._data_provider = DataSetProvider(self._config)

    def _get_dataset(
        self, stage: Stages
    ) -> Union[DataSet, Tuple[DataSet, Optional[DataSet]]]:
        total_time = time.time()

        self._logger.log_line(
            f"\nloading dataset {self._config.data_set.name.capitalize()} for {stage.name.capitalize()} ..."
        )
        data_set = self._data_provider.load_data_set(stage)

        if stage == Stages.TRAINING:
            self._logger.report_elapsed_time(
                total_time,
                f"loaded {data_set[0].size()} training items, and {data_set[1].size() if data_set[1] is not None else 0} validation items.\nloading time",
                "\n",
            )
        elif stage == Stages.TESTING:
            self._logger.report_elapsed_time(
                total_time,
                f"loaded {data_set.size()} testing items.\nloading time",
                "\n",
            )

        return data_set

    def _save_model_variables(self, epoch_index: int = 0) -> None:
        # TF checkpoints dose not work well with custom models, see
        # https://github.com/tensorflow/tensorflow/issues/33150
        self._logger.log_line("\nsaving the model ...")

        for variable in self._model.vars.to_list():
            np.save(
                self._config.model_variables_saving_path.deeper(str(epoch_index)).join(
                    variable._shared_name
                ),
                variable.numpy(),
            )

        self._logger.log_line("model variables are saved successfully.\n")

    def _load_model_variables(self, variables_path: Optional[Pather]) -> None:
        # TF checkpoints dose not work well with custom models, see
        # https://github.com/tensorflow/tensorflow/issues/33150
        self._logger.log_line(
            f"\nloading saved model from {variables_path.full_path} ..."
        )

        for variable in self._model.vars.to_list():
            array = np.load(variables_path.join(variable._shared_name, "npy"))
            variable.assign(tf.convert_to_tensor(array))

        self._logger.log_line("model variables are loaded successfully.")

    def _epochs_loop(
        self, train_dataset: DataSet, validation_dataset: Optional[DataSet]
    ) -> None:
        """
        A help function, that controls the epochs in the learning process,
        including all logging and reporting functionality.
        """

        self._logger.report_stage_start(Stages.TRAINING)

        total_time = time.time()

        def training_batch_end_callback(batch: Batch, is_final_batch: bool) -> None:
            batch.model_gradients, batch.model_variables = (
                self._model.backward_propagate(is_final_batch, is_final_batch)
            )

            if self._config.use_exponential_decay_training:
                self._current_temperature = min(
                    self._current_temperature + self._config.temperature_increment_step,
                    self._config.max_temperature,
                )
            else:
                self._current_temperature = None

        for epoch_index in range(
            self._config.start_from_epoch, self._config.training_epochs_count
        ):
            # === training
            epoch_training_time = time.time()

            if self._config.use_parametrized_step_size_reduction:
                if (
                    epoch_index
                    % self._config.parametrized_step_size_reduction_starting_epoch
                    == 0
                ):
                    self._config.optimizer.learning_rate = (
                        self._config.optimizer.learning_rate
                        * self._config.parametrized_step_size_reduction_factor
                    )

            self._logger.report_epoch_start(
                epoch_index,
                0 if self._current_temperature is None else self._current_temperature,
                self._config.optimizer.learning_rate.numpy(),
            )

            self._batches_loop(
                epoch_index,
                train_dataset,
                batch_end_callback=training_batch_end_callback,
            )

            self._logger.report_elapsed_time(epoch_training_time, "epoch training time")

            # self._save_checkpoint(epoch_index)

            # === Validate
            if self._config.validate_data:
                if validation_dataset is not None:
                    self._logger.report_stage_start(Stages.VALIDATION)

                    epoch_validating_time = time.time()

                    self._batches_loop(
                        epoch_index, validation_dataset, False, False, None
                    )

                    self._logger.report_elapsed_time(
                        epoch_validating_time, "epoch validation time"
                    )

            # save model variables
            if (
                (epoch_index + 1) % self._config.save_model_variables_every_epochs == 0
            ) or (
                epoch_index + 1
                >= (
                    self._config.training_epochs_count
                    - self._config.save_model_variables_for_last_epochs
                )
            ):
                self._save_model_variables(epoch_index + 1)

            if self._config.save_losses_to_csv:
                train_dataset.save_losses()
                if self._config.validate_data:
                    validation_dataset.save_losses()

            self._logger.report_epoch_end(total_time, epoch_index)

    def _forward_propagate(
        self,
        batch: Batch,
        use_recurrent_refinement: bool,
        use_per_image_recurrent_refinement: bool,
        stage: Stages,
    ) -> List[List[RecurrentRefinementData]]:
        """
        A wrapper over the models forward propagation, that will also update the
        batch object, and can use recurrent refinement.
        """
        is_learning = stage == Stages.TRAINING

        if use_recurrent_refinement:
            if is_learning:
                raise ValueError
            else:
                if use_per_image_recurrent_refinement:
                    (all_data, first, best) = per_image_recurrent_refinement(
                        batch.fixed_images,
                        batch.moving_images,
                        extra_loss_function_args=[
                            batch.ground_truth_displacements,
                            self._current_temperature,
                        ],
                        max_recurrent_refinement_iterations=self._config.max_recurrent_refinement_iterations,
                        model=self._model,
                    )

                    batch.pipeline_data = best.pipeline_data
                    batch.losses_storage = best.extras_from_loss_function[0]

                    self._logger.report_refinement(first, best)

                    return all_data
                else:
                    data = per_batch_recurrent_refinement(
                        batch.fixed_images,
                        batch.moving_images,
                        extra_loss_function_args=[
                            batch.ground_truth_displacements,
                            self._current_temperature,
                        ],
                        max_recurrent_refinement_iterations=self._config.max_recurrent_refinement_iterations,
                        model=self._model,
                    )

                    batch.pipeline_data = data[-1].pipeline_data
                    batch.losses_storage = data[-1].extras_from_loss_function[0]

                    self._logger.report_refinement(data[0], data[-1])

                    return [data]
        else:
            batch.pipeline_data, loss, extra = self._model.forward_propagate(
                batch.fixed_images,
                batch.moving_images,
                is_learning=is_learning,
                extra_loss_function_args=[
                    batch.ground_truth_displacements,
                    self._current_temperature,
                ],
            )

            batch.losses_storage = extra[0]

            std = tf.math.reduce_std(
                batch.fixed_images
                - batch.pipeline_data.final_variation_unit().registered_images
            ).numpy()

            return [
                [
                    RecurrentRefinementData(
                        batch.pipeline_data, loss.numpy(), std, extra, 1
                    )
                ]
            ]

    def _batches_loop(
        self,
        epoch_index: int,
        data_set: DataSet,
        use_recurrent_refinement: bool = False,
        use_per_image_recurrent_refinement: bool = False,
        batch_end_callback: Callable[[Batch, bool], None] = None,
    ) -> Batch:
        """
        Main batch loop logic, includes back propagation, profiling TF, saving
        metrics, and reporting.
        """
        starting_time = time.time()
        self._logger.report_batch_progress(data_set, starting_time)
        batches_iterator = data_set.batches_iterator()

        while not data_set.is_final_batch:
            # According to TF documentation, tracing should start before generating
            # batch data
            if (
                self._config.batches_range_to_profile[0]
                <= self._batches_processed
                <= self._config.batches_range_to_profile[1]
            ):
                if self._batches_processed == self._config.batches_range_to_profile[0]:
                    tf.profiler.experimental.start(self._config.profiling_dir)

                context = tf.profiler.experimental.Trace(
                    "train", step_num=data_set.current_batch_index, _r=1
                )
            else:
                if (
                    self._batches_processed
                    == self._config.batches_range_to_profile[1] + 1
                ):
                    tf.profiler.experimental.stop()
                context = nullcontext()

            # create a conditional `with` context for profiling
            # see https://stackoverflow.com/questions/27803059/conditional-with-statement-in-python
            with context:
                batch: Batch = next(batches_iterator)

                self._forward_propagate(
                    batch,
                    use_recurrent_refinement,
                    use_per_image_recurrent_refinement,
                    data_set.stage,
                )

                data_set.append_batch_data(batch)

                if batch_end_callback is not None:
                    batch_end_callback(batch, data_set.is_final_batch)

                if (
                    data_set.current_batch_index
                ) % self._config.report_batches_progress_every == 0:
                    self._logger.report_batch_progress(data_set, starting_time)

            self._batches_processed += 1

        # report the end of batches loop if needed
        if (
            data_set.current_batch_index
        ) % self._config.report_batches_progress_every != 0:
            self._logger.report_batch_progress(data_set, starting_time)

        self._report_batches_loop_end(epoch_index, data_set)

        # force end of sequence/stopIteration exception, to reset iterator
        try:
            next(batches_iterator)
        except Exception:
            pass

    def _report_batches_loop_end(self, epoch_index: int, data_set: DataSet) -> None:
        """
        A help function that will perform logging to the file and tensorboard
        once the batches loop is finished.
        """
        self._logger.report_batch_loop_end(epoch_index, data_set)

        if self._config.tensorboard_logging:
            # this is a time consuming process, this why we wanted to measure it.
            reporting_time = time.time()

            self._logger.log_line(
                "generating and writing tensorboard image previews ..."
            )

            self._board.report_to_tensorboard(epoch_index, data_set)

            self._logger.report_elapsed_time(
                reporting_time, "reports generating time", "\n"
            )

    def _init_model(self) -> None:
        """
        A help function, that will initialize the model and the
        optimizer, or loads it from saved checkpoints, according
        to the configurations.

        Note
        ----
        - If the model already initialized, nothing will happen.
        """
        self._batches_processed = 0

        if self._config.use_exponential_decay_training:
            self._current_temperature = self._config.initial_temperature
        else:
            self._current_temperature = None

        if self._model is not None:
            return

        if self._config.surrogate_loss_function == SurrogateLossFunction.LAST_PYRAMID:
            from source.framework.settings.surrogate_loss_functions.finest_pyramid import (
                FinestPyramidLoss,
            )

            self._config.surrogate_loss_function_instance = FinestPyramidLoss(
                self._config
            )

        elif (
            self._config.surrogate_loss_function
            == SurrogateLossFunction.PER_PYRAMID_LAST_VARIATIONAL_UNIT
        ):
            from source.framework.settings.surrogate_loss_functions.per_pyramids_last_vu import (
                PerPyramidsLastVULoss,
            )

            self._config.surrogate_loss_function_instance = PerPyramidsLastVULoss(
                self._config
            )

        else:
            raise Exception("Unsupported loss function.")

        # Initialize Variational Network:
        self._model = DIRVNet(self._config)

        self._logger.report_model(self._model)

    # endregion

    # region Public Functions

    def single_infer(
        self,
        variables_path_directory: str,
        fixed_images: tf.Tensor,
        moving_images: tf.Tensor,
        ground_truth_displacements: tf.Tensor,
        use_recurrent_refinement: bool,
        use_per_image_recurrent_refinement: bool,
    ) -> List[List[RecurrentRefinementData]]:
        """
        Performs inference on just a single tensor.

        Notes
        -----
        - This function will not perform any tensorboard logging.
        - This function will not perform any pre-processing.
        - You can pass anything for `ground_truth_displacements`, if you do not
          care about the reported loss value.

        Returns
        -------
        - Lists of `RecurrentRefinementData` objects, per iteration for each
          image/batch.
        """
        self._logger.do_not_log = True

        self._init_model()

        self._load_model_variables(Pather(variables_path_directory))

        # create a virtual batch
        batch = Batch(
            self._config, SyntheticFieldsProvider(self._config, Stages.TESTING)
        )

        batch.fixed_images = fixed_images
        batch.moving_images = moving_images
        batch.ground_truth_displacements = ground_truth_displacements

        result = self._forward_propagate(
            batch,
            use_recurrent_refinement,
            use_per_image_recurrent_refinement,
            Stages.TESTING,
        )

        self._logger.do_not_log = False

        return result

    def train(self) -> None:
        """
        Loads the data and starts the learning process, according to the
        configuration provided.
        """
        train_dataset, validation_dataset = self._get_dataset(Stages.TRAINING)

        self._logger.report_data_set(train_dataset, "Train")

        if self._config.validate_data:
            self._logger.report_data_set(validation_dataset, "Validation")

        if train_dataset is not None:
            starting_time = time.time()

            self._init_model()
            self._epochs_loop(train_dataset, validation_dataset)

            if self._config.validate_data:
                self._logger.report_stage_end(
                    [
                        train_dataset,
                        validation_dataset,
                    ],
                    starting_time,
                )
            else:
                self._logger.report_stage_end([train_dataset], starting_time)
        else:
            raise Exception("No training data is provided.")

    def test(
        self,
        use_recurrent_refinement: bool = False,
        use_per_image_recurrent_refinement: bool = False,
        model_variables_directory: Optional[str] = None,
    ) -> None:
        """
        Infers the model on the test data.

        Notes
        -----
        - if `model_variables_directory` is not provided, we assume that the
          in-memory model variables should be used, i.e. this function should be
          called without an argument only immediately after training in the same
          process.
        """
        # TODO: if test called after learning, or separately, will result in
        # different results, since seed is not reset, fix this
        # self._prepare_shape_legit(False)
        test_dataset = self._get_dataset(Stages.TESTING)

        self._logger.report_data_set(test_dataset, "Test")

        if test_dataset is not None:
            self._init_model()

            if model_variables_directory is not None:
                self._load_model_variables(Pather(model_variables_directory))

            self._logger.report_stage_start(Stages.TESTING)

            starting_time = time.time()

            self._batches_loop(
                0,
                test_dataset,
                use_recurrent_refinement,
                use_per_image_recurrent_refinement,
            )

            if self._config.save_losses_to_csv:
                test_dataset.save_losses()

            self._logger.report_stage_end([test_dataset], starting_time)
