from typing import List
from numpy.typing import NDArray

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from source.framework.settings.whole_config import WholeConfig

from source.framework.tools.general import (
    block_mean,
    discrete_curve_integral,
    calc_statistics,
)
from source.framework.tools.plotting_helper import (
    grid_figure,
    heat_map_drawer,
    annotated_heat_map_drawer,
    vector_field_drawer,
    histogram_drawer,
    line_plot_drawer,
    setup_grid_figure_color_bar,
    figure_to_tensorboard_image,
)

from source.framework.main.data_management.data_set import Stages
from source.framework.main.reporting.generators.preview_data import PreviewData


class ImagesPreviewsGenerator:
    """
    Purpose
    -------
    - Various functions to generate tensors of images cells (that provided as an
      array of plots), so that it can be used in Tensorboard.

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(self, config: WholeConfig) -> None:
        self._config: WholeConfig = config

    def _formatted_pixel(self, value: float) -> str:
        if value < 0:
            return f"{value:04.0f}"
        else:
            return f"{value:03.0f}"

    def _formatted_field(self, value: float) -> str:
        if value < 0:
            return f"{value:06.2f}"
        else:
            return f"{value:05.2f}"

    def _array_stats_str(self, array: NDArray, is_field: bool) -> str:
        stats = calc_statistics(array * (1 if is_field else 255))

        if is_field:
            mean_format = self._formatted_field(stats.mean)
            std_format = self._formatted_field(stats.standard_deviation)
            mx_format = self._formatted_field(stats.maximum)
            mn_format = self._formatted_field(stats.minimum)
        else:
            mean_format = self._formatted_pixel(stats.mean)
            std_format = self._formatted_pixel(stats.standard_deviation)
            mx_format = self._formatted_pixel(stats.maximum)
            mn_format = self._formatted_pixel(stats.minimum)

        return f"{mean_format} ± {std_format}\n[{mn_format}, {mx_format}]"

    def images_heatmap_preview(self, data: PreviewData) -> tf.Tensor:
        def drawer(y: int, x: int) -> plt.Artist:
            if isinstance(data.color_map_names, np.ndarray):
                color_map = data.color_map_names[y, x]
            else:
                color_map = data.color_map_names

            if isinstance(data.map_ranges, np.ndarray):
                ranges = data.map_ranges[y, x]
            else:
                ranges = data.map_ranges

            stats_box: bool = False
            if isinstance(data.show_stats_boxes, np.ndarray):
                stats_box = data.show_stats_boxes[y, x]
            else:
                stats_box = data.show_stats_boxes

            stats_box = stats_box and self._config.show_stats_box

            if stats_box:
                return annotated_heat_map_drawer(
                    data.grid_data[y, x],
                    box_text=self._array_stats_str(
                        data.grid_data[y, x], data.is_fields_stats
                    ),
                    title=data.titles[y, x],
                    display_grid=False,
                    display_axis=False,
                    title_font_size=self._config.image_preview_title_font_size,
                    color_map_name=color_map,
                    color_map_min_val=ranges[0],
                    color_map_max_val=ranges[1],
                    box_font_size=self._config.stats_box_font_size,
                    box_text_color=self._config.stats_box_font_color,
                    box_background_color=self._config.stats_box_background_color,
                )
            else:
                return heat_map_drawer(
                    data.grid_data[y, x],
                    title=data.titles[y, x],
                    display_grid=False,
                    display_axis=False,
                    title_font_size=self._config.image_preview_title_font_size,
                    color_map_name=color_map,
                    color_map_min_val=ranges[0],
                    color_map_max_val=ranges[1],
                )

        figure, drawn_images = grid_figure(
            data.grid_data.shape,
            drawer,
            self._config.image_preview_cell_size_inches,
        )

        if data.show_color_bar:
            setup_grid_figure_color_bar(
                figure,
                data.grid_data.shape[1],
                drawn_images[0, 0],
                data.map_ranges[0],
                data.map_ranges[1],
            )

        return figure_to_tensorboard_image(figure)

    def vector_fields_preview(self, data: PreviewData) -> tf.Tensor:
        rows = len(data.titles)
        columns = len(data.titles[0])
        figure, drawn_images = grid_figure(
            [rows, columns],
            lambda y, x: vector_field_drawer(
                data.grid_data[0][y][x],
                data.grid_data[1][y][x],
                data.titles[y][x],
                False,
                False,
                self._config.image_preview_title_font_size,
                data.color_map_names,
                data.map_ranges[1],
            ),
            self._config.vector_fields_logging_cell_size_inches,
        )

        setup_grid_figure_color_bar(
            figure, columns, drawn_images[0, 0], 0, data.map_ranges[1]
        )

        return figure_to_tensorboard_image(figure)

    def total_loss_preview(
        self, initial_losses: List, losses: List, mean_block_size: int, stage: Stages
    ) -> tf.Tensor:
        if stage == Stages.TRAINING:
            losses = block_mean(losses, mean_block_size)
        # insert the initial losses at the beginning
        # we do not want initial values to be averaged
        losses = [np.mean(initial_losses)] + losses

        def drawer(y: int, x: int) -> plt.Artist:
            return line_plot_drawer(
                [losses],
                "",
                None,
                True,
                True,
                self._config.image_preview_title_font_size,
                self._config.training_epochs_count,
            )

        figure, _ = grid_figure(
            [1, 1], drawer, self._config.potentials_logging_cell_size_inches
        )

        return figure_to_tensorboard_image(figure)

    def histograms_preview(
        self, data_list: List, crop_to_centre: bool, columns_count: int = 2
    ) -> tf.Tensor:
        titles = [
            "Data Term Filters",
            "Data Term Activations",
            "Regularization Term Filters",
            "Regularization Term Activations",
            "Learning Rates",
            "Momentum Optimization Betas",
        ]

        output_data = []

        ratio = self._config.show_histogram_of_smallest_gradients_percentage / 100
        size = len(data_list)

        for data in data_list:
            data = data.flatten()

            if crop_to_centre and len(data) > 0:  # to sense of cropping small sets
                data = np.sort(data)
                data_count_to_skip = int(1 / 2 * data.shape[0] * (1 - ratio))
                data = data[data_count_to_skip : len(data) - data_count_to_skip]

            output_data.append(data)

        if size % columns_count != 0:
            output_data += [None]

        output_data = np.array(output_data, dtype=object).reshape([-1, columns_count])
        titles = np.array(titles).reshape([-1, columns_count])

        def drawer(y: int, x: int) -> plt.Artist:
            if y == 2 and x == 0:  # learnable step sizes
                line_plot_drawer(
                    [output_data[y, x]],
                    titles[y, x],
                    None,
                    False,
                    True,
                    self._config.image_preview_title_font_size,
                )
            else:
                histogram_drawer(
                    output_data[y, x],
                    self._config.histograms_bins_count,
                    titles[y, x],
                    False,
                    self._config.image_preview_title_font_size,
                    self._config.histograms_image_alpha,
                )

        figure, _ = grid_figure(
            output_data.shape, drawer, self._config.histograms_logging_cell_size_inches
        )

        return figure_to_tensorboard_image(figure)

    def potential_functions_preview(
        self, weights: NDArray, integrate: bool
    ) -> tf.Tensor:
        output_data = np.zeros(self._config.pyramid_levels_count, dtype=object)
        titles = np.zeros(self._config.pyramid_levels_count, dtype=object)
        shared_x_data: List[float] = []

        if integrate:
            shared_x_data = np.linspace(
                self._config.potential_functions_definition_region[0],
                self._config.potential_functions_definition_region[1],
                self._config.potentials_parameters_count,
            ).tolist()
        else:
            shared_x_data = np.linspace(
                0,
                self._config.potentials_parameters_count - 1,
                self._config.potentials_parameters_count,
            ).tolist()

        for level_index in range(self._config.pyramid_levels_count):
            plot_data = []

            for filter_index in range(self._config.convolutional_kernels_count):
                data = weights[
                    level_index, -1, 0, :, filter_index
                ]  # we plotting only first channel
                if integrate:
                    data = discrete_curve_integral(
                        y=data,
                        x_from=self._config.potential_functions_definition_region[0],
                        x_to=self._config.potential_functions_definition_region[1],
                    )

                plot_data.append(data)

            output_data[level_index] = plot_data
            titles[level_index] = f"$\ell=${level_index + 1}"

        output_data = output_data.reshape(-1, 1)
        titles = titles.reshape(-1, 1)

        def drawer(y: int, x: int) -> plt.Artist:
            return line_plot_drawer(
                data_list=[output_data[y, x]][0],
                title=titles[y, x],
                labels_list=None,
                display_grid=False,
                show_axis=True,
                title_font_size=self._config.image_preview_title_font_size,
                shared_x_data=shared_x_data,
            )

        figure, _ = grid_figure(
            grid_shape=output_data.shape,
            cell_drawers=drawer,
            cell_size_inches=self._config.potentials_logging_cell_size_inches,
        )

        return figure_to_tensorboard_image(figure)

    def losses_preview(self, epoch_index: int, data: PreviewData) -> tf.Tensor:
        def drawer(y: int, x: int) -> plt.Artist:
            if x == 0:
                return line_plot_drawer(
                    data_list=data.grid_data[y, x][0],
                    title=data.titles[y, x],
                    labels_list=data.grid_data[y, x][1],
                    display_grid=False,
                    show_axis=True,
                    title_font_size=self._config.image_preview_title_font_size,
                    max_x=epoch_index,
                )
            else:
                return line_plot_drawer(
                    data_list=data.grid_data[y, x],
                    title=data.titles[y, x],
                    labels_list=None,
                    display_grid=True,
                    show_axis=True,
                    title_font_size=self._config.image_preview_title_font_size,
                    max_x=epoch_index,
                )

        figure, _ = grid_figure(
            grid_shape=data.grid_data.shape,
            cell_drawers=drawer,
            cell_size_inches=self._config.potentials_logging_cell_size_inches,
        )

        return figure_to_tensorboard_image(figure)
