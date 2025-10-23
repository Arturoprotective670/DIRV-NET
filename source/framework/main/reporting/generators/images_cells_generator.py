from typing import Any, List, Tuple
from numpy.typing import NDArray

import numpy as np
import tensorflow as tf

from source.model.main.pipe_line.pipe_line_data import VariationalUnitData
from source.framework.tools.general import block_mean

from source.framework.settings.whole_config import WholeConfig

from source.framework.main.data_management.batch import Batch
from source.framework.main.data_management.data_set import DataSet, Stages
from source.framework.main.reporting.generators.preview_data import PreviewData


class ImagesCellsGenerator:
    """
    Purpose
    -------
    - Various functions to generate cells/grid of images as an array, to be used
      latter on to generate previews.

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(self, config: WholeConfig) -> None:
        self._config: WholeConfig = config

    # region === Helper functions

    def _crop_into_list(self, data: tf.Tensor, count: int) -> List:
        """
        Converts the last `count` rows of a `Tensor` into a list of
        `np.array`'s.
        """
        return [data.numpy()[i, ...] for i in range(count)]

    def _get_displacements_range(
        self, displacements_array: NDArray[Any]
    ) -> Tuple[float, float]:
        """
        Calcs the max and min values in the displacements field. Needed to calc
        and draw color bar.
        """
        # this approach is needed because the input is an array of objects,
        # where the objects are arrays by themselves
        shape = displacements_array.shape

        maxes = []
        minis = []

        for h in range(shape[0]):
            for w in range(shape[1]):
                maxes.append(np.max(displacements_array[h, w]))
                minis.append(np.min(displacements_array[h, w]))

        mn = min(minis)
        mx = max(maxes)

        # we may have min != - max, so to force color map be symmetrical, and
        # the middle of the map zero, we need to make above min, max be
        # symmetrically spaced from center, so we take the biggest value.
        val = max(mx, abs(mn))

        return -val, val

    def _displacements_grid_max(self, displacements_gird: NDArray) -> float:
        xes = displacements_gird[0]
        yas = displacements_gird[1]

        lengths = []

        for h in range(len(xes)):
            for w in range(len(xes[0])):
                lengths.append(xes[h][w] ** 2 + yas[h][w] ** 2)

        return np.sqrt(np.max(lengths))

    def _first_2d_image(self, source: tf.Tensor):
        source = source[0, ...]

        if tf.rank(source) > 3:
            middle = tf.shape(source)[-2].numpy() // 2
            source = source[..., middle, :]

        return source.numpy()

    def _per_pyramid_loss_generator(
        self, initial_losses: List, losses: List, stage: Stages
    ):
        initial_losses = np.array(initial_losses)
        initial_losses = np.mean(initial_losses, axis=0)
        losses = np.array(losses)

        average_every = self._config.plotting_training_average_losses_every_batches

        plot_data = []
        labels = []

        for pyramid_index in range(losses.shape[1]):
            loss = losses[:, pyramid_index]
            if stage == Stages.TRAINING:
                loss = block_mean(list(loss), average_every)
            # else:
            #     loss = block_mean(list(loss), average_every)

            # initial loss should not be averaged
            loss = [initial_losses[pyramid_index]] + loss
            plot_data.append(loss)

            labels.append(f"$\ell=${pyramid_index + 1}")

        return plot_data, labels

    def _block_averaged_loss_generator(
        self, initial_losses: List, losses: List, mean_block_size: int, stage: Stages
    ):
        if stage == Stages.TRAINING:
            losses = block_mean(losses, mean_block_size)
        # insert the initial losses at the beginning
        # we do not want initial values to be averaged
        return [[np.mean(initial_losses)] + losses]

    # endregion

    # region === Public Generators

    def pyramid_images_cells_generator(self, batch: Batch) -> PreviewData:
        units_count = self._config.unfoldings_count
        levels_count = self._config.pyramid_levels_count

        # we need dtype = object to be able to assign array to an array element
        output = np.zeros([levels_count, units_count + 2], dtype=object)
        titles = np.zeros_like(output)
        color_maps = np.zeros_like(output)
        map_ranges = np.zeros_like(output)
        stats_box = np.zeros_like(output, dtype=bool)

        for pl_index, pl_data in enumerate(batch.pipeline_data.pyramids):
            fixed_images = self._first_2d_image(
                pl_data.variational_units[0].fixed_images
            )

            # ground truth is in the end
            y = pl_index
            x = units_count
            output[y, x] = fixed_images[..., 0]
            titles[y, x] = f"GT. $\ell=${pl_index + 1}, "
            color_maps[y, x] = self._config.images_preview_color_map
            map_ranges[y, x] = (0, 1)

            for vu_index, vu_data in enumerate(pl_data.variational_units):
                registered_images = self._first_2d_image(vu_data.registered_images)

                y = pl_index
                x = vu_index
                output[y, x] = registered_images[..., 0]
                titles[y, x] = f"PD. $\ell=${pl_index + 1}, $u=${vu_index + 1}"
                color_maps[y, x] = self._config.images_preview_color_map
                map_ranges[y, x] = (0, 1)

                if vu_index == units_count - 1:
                    last_unit_registration_error = fixed_images - registered_images
                    x = units_count + 1
                    output[y, x] = last_unit_registration_error[..., 0]
                    titles[y, x] = f"$\Delta$. $\ell=${pl_index + 1}"
                    color_maps[y, x] = self._config.images_diff_preview_color_map
                    map_ranges[y, x] = (-1, 1)
                    stats_box[y, x] = True

        return PreviewData(
            grid_data=output,
            titles=np.array(titles, dtype=str),
            map_ranges=map_ranges,
            color_map_names=color_maps,
            show_stats_boxes=stats_box,
            show_color_bar=False,
            is_fields_stats=False,
        )

    def pyramid_displacements_cells_generator(self, batch: Batch) -> PreviewData:
        """
        Generates a grid of images, the rows of which, are the X,Y and maybe Z
        coordinates, of the predicted displacement fields of every
        variational unit, then the downsampled ground truth displacement
        fields.
        Those rows are repeated for every pyramid.
        Also an array of strings, that represents the title of every grid image,
        is returned.
        """

        rank = batch.shape.core_rank_int
        units_count = self._config.unfoldings_count
        levels_count = self._config.pyramid_levels_count

        # we need dtype = object to be able to assign arrays as an array element
        output = np.zeros([rank * levels_count, units_count + 2], dtype=object)
        titles = np.zeros_like(output)
        stats_box = np.zeros_like(output, dtype=bool)

        axis_names = ["X", "Y", "Z"]

        for pl_index, pl_data in enumerate(batch.pipeline_data.pyramids):
            gt_displacements = self._first_2d_image(
                pl_data.variational_units[0].ground_truth_displacements
            )

            for dim in range(rank):
                y = pl_index * rank + dim
                x = units_count
                output[y, x] = gt_displacements[..., dim]
                titles[y, x] = f"GT. $\ell=${pl_index + 1}, {axis_names[dim]}"

            for vu_index, vu_data in enumerate(pl_data.variational_units):
                predicted_displacements = self._first_2d_image(
                    vu_data.predicted_displacements
                )
                predicted_displacements_error = (
                    gt_displacements - predicted_displacements
                )

                for dim in range(rank):
                    y = pl_index * rank + dim
                    x = vu_index
                    output[y, x] = predicted_displacements[..., dim]
                    titles[y, x] = (
                        f"PD. $\ell=${pl_index + 1}, $u=${vu_index + 1}, {axis_names[dim]}"
                    )

                    if vu_index == units_count - 1:
                        x = units_count + 1
                        output[y, x] = predicted_displacements_error[..., dim]
                        titles[y, x] = (
                            f"$\Delta$. $\ell=${pl_index + 1}, {axis_names[dim]}"
                        )
                        stats_box[y, x] = True

        return PreviewData(
            grid_data=output,
            titles=np.array(titles, dtype=str),
            map_ranges=self._get_displacements_range(output),
            color_map_names=self._config.displacements_preview_color_map,
            show_stats_boxes=stats_box,
            show_color_bar=True,
            is_fields_stats=True,
        )

    def batch_images_cells_generator(
        self, batch: Batch, rendering_images_count: int
    ) -> PreviewData:
        """
        Prepares a dictionary of images for `tensorboard` preview.
        """

        output = np.zeros([8, rendering_images_count], dtype=object)
        titles = np.zeros_like(output)
        color_maps = np.zeros_like(output)
        ranges = np.zeros_like(output)
        stats_box = np.zeros_like(output, dtype=bool)

        fixed_images = batch.fixed_images[..., 0]
        moving_images = batch.moving_images[..., 0]
        registered_images = (
            batch.pipeline_data.final_variation_unit().registered_images[..., 0]
        )

        output[0, :] = self._crop_into_list(fixed_images, rendering_images_count)
        titles[0, :] = [f"Fixed Image ({i + 1})" for i in range(rendering_images_count)]
        color_maps[0, :] = [
            self._config.images_preview_color_map for _ in range(rendering_images_count)
        ]
        ranges[0, :] = [[0, 1] for _ in range(rendering_images_count)]

        output[1, :] = self._crop_into_list(moving_images, rendering_images_count)
        titles[1, :] = [
            f"Moving Image ({i + 1})" for i in range(rendering_images_count)
        ]
        color_maps[1, :] = [
            self._config.images_preview_color_map for _ in range(rendering_images_count)
        ]
        ranges[1, :] = [[0, 1] for _ in range(rendering_images_count)]

        output[2, :] = self._crop_into_list(registered_images, rendering_images_count)
        titles[2, :] = [
            f"PD. Registered Image ({i + 1})" for i in range(rendering_images_count)
        ]
        color_maps[2, :] = [
            self._config.images_preview_color_map for _ in range(rendering_images_count)
        ]
        ranges[2, :] = [[0, 1] for _ in range(rendering_images_count)]

        output[3, :] = self._crop_into_list(
            fixed_images - moving_images, rendering_images_count
        )
        titles[3, :] = [
            f"Fixed - Moving ({i + 1})" for i in range(rendering_images_count)
        ]
        color_maps[3, :] = [
            self._config.images_diff_preview_color_map
            for _ in range(rendering_images_count)
        ]
        ranges[3, :] = [[-1, 1] for _ in range(rendering_images_count)]
        stats_box[3, :] = True

        output[4, :] = self._crop_into_list(
            fixed_images - registered_images, rendering_images_count
        )
        titles[4, :] = [
            f"Fixed - Registered ({i + 1})" for i in range(rendering_images_count)
        ]
        color_maps[4, :] = [
            self._config.images_diff_preview_color_map
            for _ in range(rendering_images_count)
        ]
        ranges[4, :] = [[-1, 1] for _ in range(rendering_images_count)]
        stats_box[4, :] = True

        if self._config.image_preview_drawn_grid_lines_count > 1:
            if self._config.swap_between_fixed_moving_images:
                # see interpolated moving images using the inverted fields, to check
                # the quality of inversion
                output[5, :] = self._crop_into_list(
                    batch.grided_moving_images()[..., 0], rendering_images_count
                )
                suffix = (
                    "Grid "
                    if self._config.image_preview_drawn_grid_lines_count > 1
                    else ""
                )
                titles[5, :] = [
                    f"Int. Moving {suffix}({i + 1})"
                    for i in range(rendering_images_count)
                ]
                color_maps[5, :] = [
                    self._config.images_preview_color_map
                    for _ in range(rendering_images_count)
                ]
                ranges[5, :] = [[0, 1] for _ in range(rendering_images_count)]

            output[6, :] = self._crop_into_list(
                batch.grided_registered_images(True)[..., 0], rendering_images_count
            )
            titles[6, :] = [
                f"GT. Registered Grid ({i + 1})" for i in range(rendering_images_count)
            ]
            color_maps[6, :] = [
                self._config.images_preview_color_map
                for _ in range(rendering_images_count)
            ]
            ranges[6, :] = [[0, 1] for _ in range(rendering_images_count)]

            output[7, :] = self._crop_into_list(
                batch.grided_registered_images(False)[..., 0], rendering_images_count
            )
            titles[7, :] = [
                f"PD. Registered Grid ({i + 1})" for i in range(rendering_images_count)
            ]
            color_maps[7, :] = [
                self._config.images_preview_color_map
                for _ in range(rendering_images_count)
            ]
            ranges[7, :] = [[0, 1] for _ in range(rendering_images_count)]

        # delete un-needed rows, should be done in a reverse order
        if self._config.image_preview_drawn_grid_lines_count <= 1:
            output = np.delete(output, 7, axis=0)
            titles = np.delete(titles, 7, axis=0)
            color_maps = np.delete(color_maps, 7, axis=0)
            ranges = np.delete(ranges, 7, axis=0)
            stats_box = np.delete(stats_box, 7, axis=0)

            output = np.delete(output, 6, axis=0)
            titles = np.delete(titles, 6, axis=0)
            color_maps = np.delete(color_maps, 6, axis=0)
            ranges = np.delete(ranges, 6, axis=0)
            stats_box = np.delete(stats_box, 6, axis=0)

            output = np.delete(output, 5, axis=0)
            titles = np.delete(titles, 5, axis=0)
            color_maps = np.delete(color_maps, 5, axis=0)
            ranges = np.delete(ranges, 5, axis=0)
            stats_box = np.delete(stats_box, 5, axis=0)

        if (
            not self._config.swap_between_fixed_moving_images
            and self._config.image_preview_drawn_grid_lines_count > 1
        ):
            output = np.delete(output, 5, axis=0)
            titles = np.delete(titles, 5, axis=0)
            color_maps = np.delete(color_maps, 5, axis=0)
            ranges = np.delete(ranges, 5, axis=0)
            stats_box = np.delete(stats_box, 5, axis=0)

        return PreviewData(
            grid_data=output,
            titles=np.array(titles, dtype=str),
            map_ranges=ranges,
            color_map_names=color_maps,
            show_stats_boxes=stats_box,
            show_color_bar=False,
            is_fields_stats=False,
        )

    def batch_displacements_cells_generator(
        self, batch: Batch, rendering_images_count: int
    ) -> PreviewData:
        output = np.zeros(
            [3 * self._config.images_rank, rendering_images_count], dtype=object
        )
        titles = np.zeros_like(output)
        stats_box = np.zeros_like(output, dtype=bool)

        axis_names = ["X", "Y", "Z"]

        final_vu_data: VariationalUnitData = None
        if self._config.images_rank == 3:
            final_vu_data = batch.slice_batch(False).pipeline_data.final_variation_unit()
        else:
            final_vu_data = batch.pipeline_data.final_variation_unit()

        gt_displacements = final_vu_data.ground_truth_displacements
        predicted_displacements = final_vu_data.predicted_displacements

        for dim in range(self._config.images_rank):
            trans = gt_displacements[..., dim]
            output[3 * dim, :] = self._crop_into_list(trans, rendering_images_count)
            titles[3 * dim, :] = [
                f"GT. {axis_names[dim]} ({i + 1})"
                for i in range(rendering_images_count)
            ]

            trans = predicted_displacements[..., dim]
            output[3 * dim + 1, :] = self._crop_into_list(trans, rendering_images_count)
            titles[3 * dim + 1, :] = [
                f"PD. {axis_names[dim]} ({i + 1})"
                for i in range(rendering_images_count)
            ]

            trans = gt_displacements[..., dim] - predicted_displacements[..., dim]
            output[3 * dim + 2, :] = self._crop_into_list(trans, rendering_images_count)
            titles[3 * dim + 2, :] = [
                f"$\Delta${axis_names[dim]} ({i + 1})"
                for i in range(rendering_images_count)
            ]
            stats_box[3 * dim + 2, :] = True

        return PreviewData(
            grid_data=output,
            titles=np.array(titles, dtype=str),
            map_ranges=self._get_displacements_range(output),
            color_map_names=self._config.displacements_preview_color_map,
            show_stats_boxes=stats_box,
            show_color_bar=True,
            is_fields_stats=True,
        )

    def batch_vector_fields_cells_generator(
        self, batch: Batch, rendering_images_count: int
    ) -> PreviewData:
        xes = []
        yas = []
        titles = []

        final_vu_data: VariationalUnitData = None
        if self._config.images_rank == 3:
            final_vu_data = batch.slice_batch().pipeline_data.final_variation_unit()
        else:
            final_vu_data = batch.pipeline_data.final_variation_unit()

        gt_displacements = final_vu_data.ground_truth_displacements
        pred_displacements = final_vu_data.predicted_displacements

        disp = gt_displacements[..., 0]
        xes.append(self._crop_into_list(disp, rendering_images_count))
        disp = gt_displacements[..., 1]
        yas.append(self._crop_into_list(disp, rendering_images_count))
        titles.append([f"GT. ({i + 1})" for i in range(rendering_images_count)])

        disp = pred_displacements[..., 0]
        xes.append(self._crop_into_list(disp, rendering_images_count))
        disp = pred_displacements[..., 1]
        yas.append(self._crop_into_list(disp, rendering_images_count))
        titles.append([f"PD. ({i + 1})" for i in range(rendering_images_count)])

        disp = gt_displacements[..., 0] - pred_displacements[..., 0]
        xes.append(self._crop_into_list(disp, rendering_images_count))
        disp = gt_displacements[..., 1] - pred_displacements[..., 1]
        yas.append(self._crop_into_list(disp, rendering_images_count))
        titles.append([f"$\Delta$ ({i + 1})" for i in range(rendering_images_count)])

        res = [xes, yas]

        return PreviewData(
            grid_data=res,
            titles=np.array(titles, dtype=str),
            map_ranges=[
                -self._displacements_grid_max(res),
                self._displacements_grid_max(res),
            ],
            color_map_names=self._config.vector_fields_images_preview_color_map,
        )

    def losses_cells_generator(self, data_set: DataSet) -> PreviewData:
        output = np.zeros([2, 2], dtype=object)
        titles = np.zeros_like(output)

        titles[0, 0] = "Per Level Displacement Fields Loss"
        output[0, 0] = self._per_pyramid_loss_generator(
            data_set.initial_losses_storage.per_batch_per_level_displacements_losses,
            data_set.losses_storage.per_batch_per_level_displacements_losses,
            data_set.stage,
        )

        titles[0, 1] = "Level Avg. Displacement Fields Loss"
        output[0, 1] = self._block_averaged_loss_generator(
            data_set.initial_losses_storage.per_batch_displacements_losses,
            data_set.losses_storage.per_batch_displacements_losses,
            self._config.plotting_training_average_losses_every_batches,
            data_set.stage,
        )

        titles[1, 0] = "Per Level Images Intensities Loss"
        output[1, 0] = self._per_pyramid_loss_generator(
            data_set.initial_losses_storage.per_batch_per_level_image_intensity_losses,
            data_set.losses_storage.per_batch_per_level_image_intensity_losses,
            data_set.stage,
        )

        titles[1, 1] = "Level Avg. Images Intensities Loss"
        output[1, 1] = self._block_averaged_loss_generator(
            data_set.initial_losses_storage.per_batch_intensity_losses,
            data_set.losses_storage.per_batch_intensity_losses,
            self._config.plotting_training_average_losses_every_batches,
            data_set.stage,
        )

        return PreviewData(grid_data=output, titles=np.array(titles, dtype=str))

    # endregion
