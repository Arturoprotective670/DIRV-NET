from __future__ import annotations
from typing import List, Optional, Callable

import numpy as np
from numpy.typing import NDArray
import tensorflow as tf

from source.framework.tools.general import StatisticsData, calc_statistics
from source.model.main.learnable_variables import LearnableVariables
from source.model.main.pipe_line.pipe_line_data import (
    PipeLineData,
    PyramidLevelData,
    VariationalUnitData,
)

from source.model.tools.legit_shape import is_legit_dimension_size
from source.model.tools.shape_break_down import ShapeBreakDown
from source.model.tools.operators.warping_operator import warp_images
from source.model.tools.operators.sampling_operator import sample

from source.framework.main.data_management.losses_storage import LossesStorage

from source.framework.settings.whole_config import WholeConfig
from source.framework.settings.synthetic_fields_provider import (
    SyntheticFieldsProvider,
)
from source.framework.settings.pre_processor import PreProcessor

from source.framework.tools.field_invertor import inverted_fields_from_flows
from source.framework.tools.pather import Pather


class Batch:
    """
    Purpose
    --------
    - Holds various batch information, like fixed, moving, registered data, and
      various fields, all in `Tensor` format. In addition to various
      functionality, like drawing grids, applying fields, loading and saving
      data from/to files.

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(
        self, config: WholeConfig, _fields_provider: SyntheticFieldsProvider
    ) -> None:
        self._config = config

        # original images will be set to fixed images, or maybe pre-processed first
        self.original_images: Optional[tf.Tensor] = None

        self.fixed_images: Optional[tf.Tensor] = None
        self.moving_images: Optional[tf.Tensor] = None

        self.ground_truth_flows: Optional[tf.Tensor] = None
        self.ground_truth_inverse_flows: Optional[tf.Tensor] = None

        self.ground_truth_displacements = None
        self.ground_truth_inverse_displacements = None

        self.losses_storage: Optional[LossesStorage] = None

        self.model_gradients : Optional[List[NDArray]] = None
        self.model_variables: Optional[LearnableVariables] = None

        self.pipeline_data: Optional[PipeLineData] = None

        self._grided_moving_images: Optional[tf.Tensor] = None
        self.shape: Optional[ShapeBreakDown] = None

        self._fields_provider = _fields_provider

        self._pre_processor: Callable[[tf.Tensor], tf.Tensor] = PreProcessor(
            config
        ).pre_process_images

    # region === Help functions

    def _slice_pipeline_data(
        self, pipeline_data: PipeLineData, change_fields_components_count: bool
    ) -> PipeLineData:
        sliced_pipeline_data = PipeLineData()

        for pl_data in pipeline_data.pyramids:
            sliced_pl_data = PyramidLevelData()

            for vu_unit in pl_data.variational_units:
                sliced_vu_data = VariationalUnitData()

                sliced_vu_data.input_displacements = self._field_slice(
                    vu_unit.input_displacements, change_fields_components_count
                )
                sliced_vu_data.input_flows = self._field_slice(
                    vu_unit.input_flows, change_fields_components_count
                )

                sliced_vu_data.fixed_images = self._images_slice(vu_unit.fixed_images)
                sliced_vu_data.moving_images = self._images_slice(vu_unit.moving_images)
                sliced_vu_data.registered_images = self._images_slice(
                    vu_unit.registered_images
                )

                sliced_vu_data.predicted_displacements = self._field_slice(
                    vu_unit.predicted_displacements, change_fields_components_count
                )
                sliced_vu_data.predicted_flows = self._field_slice(
                    vu_unit.predicted_flows, change_fields_components_count
                )

                sliced_vu_data.ground_truth_displacements = self._field_slice(
                    vu_unit.ground_truth_displacements, change_fields_components_count
                )
                sliced_vu_data.ground_truth_flow = self._field_slice(
                    vu_unit.ground_truth_flow, change_fields_components_count
                )

                sliced_pl_data.variational_units.append(sliced_vu_data)

            sliced_pipeline_data.pyramids.append(sliced_pl_data)

        return sliced_pipeline_data

    def _images_slice(self, images: Optional[tf.Tensor]) -> Optional[tf.Tensor]:
        if images is None:
            return None

        half_depth = tf.shape(images)[-2] // 2

        return images[..., half_depth, :]

    def _draw_grid_and_warp_images(
        self,
        images: tf.Tensor,
        flows: tf.Tensor,
        grided_data: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """
        Takes an images batch, and draws on it a grid, and warps/applies them
        with the provided flows.

        Notes
        -----
        - If `grided_data` is provided, then `images` is ignored, and
          `grided_data` is used instead, and the grid is not drawn on. Instead,
          the flows will be applied directly on them.
        """

        from framework.tools.plotting_helper import draw_batch_tensor_grid

        if self._config.image_preview_drawn_grid_lines_count > 1:
            if grided_data is None:
                spacing = (
                    self.shape.core_shape
                    // self._config.image_preview_drawn_grid_lines_count
                )
                grided_data = draw_batch_tensor_grid(
                    images,
                    grid_spacings=spacing,
                    grid_lines_intensity=self._config.image_preview_grid_intensity,
                )
        else:
            grided_data = images

        images, _ = warp_images(
            grided_data, flows, self._config.white_noise_flows_threshold
        )

        return images

    def _calc_legit_shape(
        self, original_shape: List[int], find_smaller_shape: bool = False
    ) -> List[int]:
        """
        Calcs the closest shape to the given one, such that it satisfies the
        `DIRV-Net` size requirements.
        """
        legit_shape = []
        step = -1 if find_smaller_shape else 1

        for dim in range(len(original_shape)):
            new_size = original_shape[dim]

            while not is_legit_dimension_size(
                new_size,
                self._config.pyramid_levels_count,
                self._config.displacements_control_points_spacings[dim],
            ):
                new_size += step

            legit_shape.append(new_size)

        return legit_shape

    def _pad_to_shape(self, source: tf.Tensor, new_shape: List) -> tf.Tensor:
        """
        Pads symmetrically (if possible) a tensor with zeros.

        Arguments
        ---------
        - `source`: The `Tensor` we want to pad.
        - `new_shape`: The new desired shape (excluding the batch and channels
        dimensions).

        Notes
        ------
        - If padding can't be done symmetrically, the extra pixels are added at
        the end of the dimension.
        """
        images_shape = tf.shape(source)[1:-1].numpy()
        paddings = []
        # no padding for the batch dimension
        paddings.append([0, 0])

        for dim in range(len(images_shape)):
            diff = new_shape[dim] - images_shape[dim]

            if diff > 1:
                paddings.append([diff // 2, diff - (diff // 2)])
            elif diff == 0 or diff == 1:
                paddings.append([0, diff])
            else:
                raise Exception("Something is wrong with padding sizes.")

        # no paddings for the channel dimension
        paddings.append([0, 0])

        return tf.pad(source, paddings)

    def _generate_ground_truth_inverse_fields(self) -> None:
        """
        - Generates the `ground_truth_inverse_displacements` and
        `ground_truth_inverse_flows`, by inverting the `ground_truth_flows`.
        - This is used when `ground_truth_inverse_flows` are not available, for
        example when the config's `Swap Trick` is enabled.
        - Mainly needed for plotting epoch's last batch results, in
        `grided_moving_images` function.
        """
        self.ground_truth_inverse_displacements, self.ground_truth_inverse_flows = (
            inverted_fields_from_flows(
                self.ground_truth_flows,
                self._config.displacements_control_points_spacings,
                self._config.inverse_field_sampling_quality,
            )
        )

    def _field_slice(
        self, field: Optional[tf.Tensor], change_fields_components_count: bool
    ) -> Optional[tf.Tensor]:
        if field is None:
            return None

        half_depth = tf.shape(field)[-2] // 2

        if change_fields_components_count:
            return field[..., half_depth, 0:2]
        else:
            return field[..., half_depth, :]

    def _dump_tensors_list(self, arrays_list, file_path: str) -> None:
        with open(file_path, "wb") as file_handle:
            for i in arrays_list:
                np.save(file_handle, arrays_list[i].numpy())

    # endregion

    # region === Public functions

    def grided_moving_images(self) -> tf.Tensor:
        """
        Generates moving images with a grid on them, by applying the current
        ground truth inverse flow fields on the current fixed images after
        drawing a grid on them.
        """

        # it could be that inverse fields are not available if swap trick is on,
        # generate them before we continue.
        if (
            self._config.swap_between_fixed_moving_images
            and self.ground_truth_inverse_flows is None
        ):
            self._generate_ground_truth_inverse_fields()

        if self._config.swap_between_fixed_moving_images:
            self._grided_moving_images = self._draw_grid_and_warp_images(
                self.fixed_images, self.ground_truth_inverse_flows
            )

        return self._grided_moving_images

    def grided_registered_images(self, get_ground_truth_results: bool) -> tf.Tensor:
        """
        Generates registered images with a grid on them, by applying the current
        flows on the current moving images after drawing a grid on them.

        Arguments
        ---------
        - `get_ground_truth_results`: if set to `True`, the grided registered
        images will be generated by applying the `ground_truth_flows`, otherwise,
        the `predicted_flows` will be applied.
        """

        if get_ground_truth_results:
            return self._draw_grid_and_warp_images(
                self.moving_images,
                self.pipeline_data.final_variation_unit().ground_truth_flows,
                self._grided_moving_images,
            )
        else:
            return self._draw_grid_and_warp_images(
                self.moving_images,
                self.pipeline_data.final_variation_unit().predicted_flows,
                self._grided_moving_images,
            )

    def initialize_batch_data(self) -> None:
        """
        Initializes/generates the following:
        - Fixed images, by pre-processing original data
        - Adapting images size by padding (Needed when we use patches, or not
        supported sizes by `DIRV-Net`).
        - Generating moving images.
        - Generating ground truth displacement and flow fields.
        - Generates inverse of above fields, depending on if `Swap Trick` is used or not.

        Notes
        -----
        - We need this as a separate function, because we need to setup data
          manually in `slice_to_2D_batch` functions.
        """
        # === first we generate fixed images
        self.fixed_images = self._pre_processor(self.original_images)

        # === perform padding if needed
        initial_images_shape = ShapeBreakDown(self.fixed_images).core_shape_list
        legit_images_shape = self._calc_legit_shape(initial_images_shape)

        if legit_images_shape != initial_images_shape:
            print(
                "\t\t\t!! Warning !!\t\t\t\nImages of the shape "
                + f"{initial_images_shape} are not suitable for this "
                + f"model, they will be padded to the shape {legit_images_shape}\n"
            )
            self.fixed_images = self._pad_to_shape(
                self.fixed_images, legit_images_shape
            )

        self.shape = ShapeBreakDown(self.fixed_images)

        # === next we generate GT flow and displacement fields
        synthetic_flows, poisoned_synthetic_flows = self._fields_provider.generate(
            self.shape
        )

        synthetic_displacements = sample(
            synthetic_flows,
            self._config.displacements_control_points_spacings,
        )

        # === next we generate moving images, the process depends on if we will
        # use swapping trick, keep in mind the the model always predicts the
        # direct flow fields, not the inverted one.
        if not self._config.swap_between_fixed_moving_images:
            # we consider above fields, to be the flow field inverse, so we can
            # use them to calculate the moving images

            self.ground_truth_inverse_flows = synthetic_flows
            self.ground_truth_inverse_displacements = synthetic_displacements

            # generate moving images from fields
            self.moving_images, _ = warp_images(
                self.fixed_images,
                poisoned_synthetic_flows,
                self._config.white_noise_flows_threshold,
            )

            # however, we need to predict the direct flow fields (see eq. 4),
            # not the inverted one, so we need to calc the inverse
            self.ground_truth_displacements, self.ground_truth_flows = (
                inverted_fields_from_flows(
                    synthetic_flows,
                    self._config.displacements_control_points_spacings,
                    self._config.inverse_field_sampling_quality,
                )
            )

            # note that in principle, one can do here the opposite, i.e. to
            # consider synthetic fields to be the direct ones, however in both
            # cases one will need to calc the inverse.
        else:
            # first, we also consider the synthetic fields to be the inverse of
            # flows, and we calculate the moving images
            self.moving_images, _ = warp_images(
                self.fixed_images,
                poisoned_synthetic_flows,
                self._config.white_noise_flows_threshold,
            )

            # now, we swap the moving with fixed images
            self.moving_images, self.fixed_images = (
                self.fixed_images,
                self.moving_images,
            )

            # by such swap, synthetic fields become regular fields, not the
            # inverted one!
            self.ground_truth_flows = synthetic_flows
            self.ground_truth_displacements = synthetic_displacements

            # by this, we managed to generate the moving images, without
            # calculating the inverse fields, and loosing quality.

            # However, we may still need to calculate the inverse fields for
            # tensorboard drawing, see `grided_moving_images` function, but none
            # of the calculations will get affected by the quality of the
            # calculated inverse of fields.

    def slice_batch(self, change_fields_components_count: bool = True) -> Batch:
        """
        - Slices a batch of 3D images in the middle, and generates a 2D batch
        object of it.
        - Used mainly for `tensorboard` drawing.
        """
        sliced_batch = Batch(self._config, self._fields_provider)

        sliced_batch.original_images = self._images_slice(self.original_images)
        sliced_batch.fixed_images = self._images_slice(self.fixed_images)
        sliced_batch.moving_images = self._images_slice(self.moving_images)

        sliced_batch.ground_truth_flows = self._field_slice(
            self.ground_truth_flows, change_fields_components_count
        )
        sliced_batch.ground_truth_displacements = self._field_slice(
            self.ground_truth_displacements, change_fields_components_count
        )
        sliced_batch.ground_truth_inverse_flows = self._field_slice(
            self.ground_truth_inverse_flows, change_fields_components_count
        )
        sliced_batch.ground_truth_inverse_displacements = self._field_slice(
            self.ground_truth_inverse_displacements, change_fields_components_count
        )

        sliced_batch.shape = ShapeBreakDown(sliced_batch.fixed_images)

        sliced_batch.pipeline_data = self._slice_pipeline_data(
            self.pipeline_data, change_fields_components_count
        )

        return sliced_batch

    def dump_data(self, directory_path: Pather, starting_index: int) -> None:
        for i in range(self.shape.batch_size_int):
            to_save = []
            to_save.append(self.fixed_images[i, ...])
            to_save.append(self.moving_images[i, ...])
            to_save.append(self.registered_images[i, ...])
            to_save.append(self.ground_truth_flows[i, ...])
            to_save.append(self.ground_truth_inverse_flows[i, ...])
            to_save.append(self.ground_truth_inverse_displacements[i, ...])
            to_save.append(self.ground_truth_displacements[i, ...])

            self._dump_tensors_list(
                to_save, directory_path.join(f"{i + starting_index:06}", "npy")
            )

    def ground_truth_flows_components_statistics(self) -> StatisticsData:
        return calc_statistics(self.ground_truth_displacements)

    def ground_truth_flows_magnitude_statistics(self) -> StatisticsData:
        return calc_statistics(tf.norm(self.ground_truth_displacements, axis=-1))


# endregion
