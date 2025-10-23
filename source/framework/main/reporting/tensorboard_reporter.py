from typing import Any, Dict, List

import tensorflow as tf

from source.framework.settings.whole_config import WholeConfig

from source.framework.main.data_management.data_set import DataSet, Stages
from source.framework.main.reporting.generators.images_cells_generator import (
    ImagesCellsGenerator,
)
from source.framework.main.reporting.generators.images_previews_generator import (
    ImagesPreviewsGenerator,
)


class TensorBoardReporter:
    """
    Purpose
    -------
    - Various functions to generate Tensorboard reports and images.

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(self, config: WholeConfig) -> None:
        self._config: WholeConfig = config

        self._grid_generator = ImagesCellsGenerator(config)
        self._preview_generator = ImagesPreviewsGenerator(config)

    # region === Help Functions

    def _dic_to_list(self, dict: Dict[str, str]) -> List:
        results = []

        for key, value in dict.items():
            if key.startswith("_") is False:  # do not print internal props
                results.append(f"<li>{key.replace('_',' ')}={value}</li>")

        return results

    def _get_config_contents(self) -> str:
        items = self._dic_to_list(vars(self._config))

        return "<ol>" + "\n".join(items) + "</ol>"

    def _get_tensorboard_writer(self, stage: Stages) -> Any:
        if stage == Stages.TRAINING:
            if hasattr(self, "_training_writer") is False:
                self._training_writer = tf.summary.create_file_writer(
                    self._config.train_log_dir
                )
            return self._training_writer

        if stage == Stages.VALIDATION:
            if hasattr(self, "_validation_writer") is False:
                self._validation_writer = tf.summary.create_file_writer(
                    self._config.validation_log_dir
                )
            return self._validation_writer

        if stage == Stages.TESTING:
            if hasattr(self, "_testing_writer") is False:
                self._testing_writer = tf.summary.create_file_writer(
                    self._config.test_log_dir
                )
            return self._testing_writer

        if stage == Stages.INFERRING:
            if hasattr(self, "_inferring_writer") is False:
                self._inferring_writer = tf.summary.create_file_writer(
                    self._config.inferring_log_dir
                )
            return self._inferring_writer

    # endregion

    def report_to_tensorboard(self, epoch_index: int, data_set: DataSet) -> None:
        stage_name = data_set.stage.name.capitalize() + " "
        batch = data_set.current_batch

        if self._config.tensorboard_logging:
            # generate the range of image indexes that will be reported
            rendering_images_count = min(
                batch.shape.batch_size_int, self._config.tensorboard_max_logged_images
            )

            if batch.shape.core_rank_int == 3:
                # we draw the z-middle slice, for that we create a sliced batch
                # replace batch with sliced version of it
                # we need this, to keep inverse trans. generation possible, when we
                # have big 3D images, otherwise, inverse fails for some internal reason
                sub_batch = batch.slice_batch()
            else:
                sub_batch = batch

            batch_images_draw_data = self._grid_generator.batch_images_cells_generator(
                sub_batch, rendering_images_count
            )
            batch_displacements_draw_data = (
                self._grid_generator.batch_displacements_cells_generator(
                    batch, rendering_images_count
                )
            )
            batch_vectors_draw_data = (
                self._grid_generator.batch_vector_fields_cells_generator(
                    batch, rendering_images_count
                )
            )
            pyramids_images_draw_data = (
                self._grid_generator.pyramid_images_cells_generator(batch)
            )
            pyramid_displacements_draw_data = (
                self._grid_generator.pyramid_displacements_cells_generator(batch)
            )

            losses_draw_data = self._grid_generator.losses_cells_generator(data_set)

            with self._get_tensorboard_writer(data_set.stage).as_default():
                epoch_index += 1

                with tf.name_scope("Last Batch Details"):
                    tf.summary.image(
                        stage_name + "images",
                        self._preview_generator.images_heatmap_preview(
                            batch_images_draw_data
                        ),
                        step=epoch_index,
                    )

                    tf.summary.image(
                        stage_name + "displacement fields",
                        self._preview_generator.images_heatmap_preview(
                            batch_displacements_draw_data
                        ),
                        step=epoch_index,
                    )

                    tf.summary.image(
                        stage_name + "displacement vector fields",
                        data=self._preview_generator.vector_fields_preview(
                            batch_vectors_draw_data
                        ),
                        step=epoch_index,
                    )

                with tf.name_scope("Last Image Details"):
                    tf.summary.image(
                        stage_name + " registered images per level per unit",
                        self._preview_generator.images_heatmap_preview(
                            pyramids_images_draw_data
                        ),
                        step=epoch_index,
                    )

                    tf.summary.image(
                        stage_name + " displacement fields per level per unit",
                        self._preview_generator.images_heatmap_preview(
                            pyramid_displacements_draw_data
                        ),
                        step=epoch_index,
                    )

                if data_set.stage == Stages.TRAINING:
                    if epoch_index == 1:
                        with tf.name_scope("Configurations"):
                            tf.summary.text(
                                "items", self._get_config_contents(), step=0
                            )

                    assert batch.model_gradients is not None

                    with tf.name_scope("Model Internals"):
                        tf.summary.image(
                            stage_name
                            + f"{self._config.show_histogram_of_smallest_gradients_percentage}% of smallest gradients histograms",
                            self._preview_generator.histograms_preview(
                                batch.model_gradients, True
                            ),
                            step=epoch_index,
                        )

                        tf.summary.image(
                            stage_name + "weights histograms",
                            self._preview_generator.histograms_preview(
                                batch.model_variables.to_list(), False
                            ),
                            step=epoch_index,
                        )

                        tf.summary.image(
                            stage_name
                            + "per level, last unit, first channel, data term potential functions",
                            self._preview_generator.potential_functions_preview(
                                batch.model_variables.data_term_potential_function_parameters, False
                            ),
                            step=epoch_index,
                        )

                        tf.summary.image(
                            stage_name
                            + "per level, last unit, first channel, data term activation functions",
                            self._preview_generator.potential_functions_preview(
                                batch.model_variables.data_term_potential_function_parameters, True
                            ),
                            step=epoch_index,
                        )

                        tf.summary.image(
                            stage_name
                            + "er level, last unit, first channel, regularization term potential functions",
                            self._preview_generator.potential_functions_preview(
                                batch.model_variables.regularization_term_potential_function_parameters, False
                            ),
                            step=epoch_index,
                        )

                        tf.summary.image(
                            stage_name
                            + "er level, last unit, first channel, regularization activation functions",
                            self._preview_generator.potential_functions_preview(
                                batch.model_variables.regularization_term_potential_function_parameters, True
                            ),
                            step=epoch_index,
                        )

                with tf.name_scope("Losses"):
                    tf.summary.image(
                        stage_name + "losses",
                        self._preview_generator.losses_preview(
                            epoch_index, losses_draw_data
                        ),
                        step=epoch_index,
                    )

                    tf.summary.image(
                        stage_name + "loss function",
                        self._preview_generator.total_loss_preview(
                            data_set.initial_losses_storage.per_image_combined_loss,
                            data_set.losses_storage.per_image_combined_loss,
                            self._config.plotting_training_average_losses_every_batches
                            * self._config.training_batch_size,
                            data_set.stage,
                        ),
                        step=epoch_index,
                    )
