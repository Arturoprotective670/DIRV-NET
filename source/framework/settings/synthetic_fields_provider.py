from typing import Callable
import numpy as np
import tensorflow as tf

from source.framework.settings.enums import SyntheticFields, Stages
from source.framework.settings.whole_config import WholeConfig

from source.framework.tools.general import (
    random_binary_choice_indexes,
    truncated_white_noise,
)
from source.model.tools.shape_break_down import ShapeBreakDown

from source.framework.tools.affine_random_fields import (
    batch_random_image_affine_matrices,
    flow_fields_from_batch_affine_matrices,
)
from source.framework.tools.non_rigid_random_fields import NonRigidRandomFields


class SyntheticFieldsProvider:
    """
    Purpose
    -------
    - Generate various synthetic fields depending of the config file settings.

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(self, config: WholeConfig, stage: Stages) -> None:
        self._config = config

        if stage == Stages.TRAINING:
            random_seed = config.global_random_seed
        elif stage == Stages.TESTING:
            random_seed = config.test_global_random_seed

        # random generator here should be a separate one, to not get interfered
        # with others, also affine and non-rigid should have separate one also
        self._affine_random_generator = np.random.default_rng(random_seed)
        self._nonrigid_random_generator = np.random.default_rng(random_seed)
        self._noise_random_generator = np.random.default_rng(random_seed)

        if (
            config.synthetic_flow_fields == SyntheticFields.NON_RIGID_FIELDS
            or config.synthetic_flow_fields == SyntheticFields.COMBINED
        ):
            self._non_rigid_fields = NonRigidRandomFields(
                config.images_rank,
                config.displacements_to_flows_resizing_up_method,
                config.non_rigid_smoothing_kernels,
                random_seed,
            )

    def generate(self, batch_shape: ShapeBreakDown) -> tf.Tensor:
        fields: tf.t = None

        if self._config.synthetic_flow_fields == SyntheticFields.AFFINE_FIELDS:
            fields = self._affaine_generator(batch_shape)
        elif self._config.synthetic_flow_fields == SyntheticFields.NON_RIGID_FIELDS:
            fields = self._non_rigid_generator(batch_shape)
        elif self._config.synthetic_flow_fields == SyntheticFields.COMBINED:
            fields = self._combined_generator(batch_shape)
        elif (
            self._config.synthetic_flow_fields == SyntheticFields.TRUNCATED_WHITE_NOISE
        ):
            return self._truncated_white_noise(batch_shape)
        else:
            raise NotImplementedError

        # randomly set some flows to be white noise
        fields = self._replace_with_white_noise(fields, batch_shape)

        poisoned_fields = self._poison_with_white_noise(
            tf.identity(fields), batch_shape
        )

        return fields, poisoned_fields

    def _replace_with_white_noise(
        self, fields: tf.Tensor, batch_shape: ShapeBreakDown
    ) -> tf.Tensor:
        false_indexes = random_binary_choice_indexes(
            batch_shape.batch_size_int,
            self._config.white_noise_flows_replacing_probability,
            self._noise_random_generator,
        )

        image_shape = [1] + batch_shape.core_shape_list + [1]
        image_shape = tf.convert_to_tensor(image_shape)
        image_shape = ShapeBreakDown(image_shape, True)

        for i in false_indexes:
            noise = self._truncated_white_noise(image_shape)
            fields = tf.tensor_scatter_nd_update(fields, indices=[[i]], updates=noise)

        return fields

    def _poison_with_white_noise(
        self, fields: tf.Tensor, batch_shape: ShapeBreakDown
    ) -> tf.Tensor:
        false_indexes = random_binary_choice_indexes(
            batch_shape.batch_size_int,
            self._config.white_noise_flows_poison_probability,
            self._noise_random_generator,
        )

        image_shape = [1] + batch_shape.core_shape_list + [1]
        image_shape = tf.convert_to_tensor(image_shape)
        image_shape = ShapeBreakDown(image_shape, True)

        for i in false_indexes:
            noise = self._truncated_white_noise(image_shape)
            field = fields[i, ...]
            field = tf.expand_dims(field, axis=0) + noise
            fields = tf.tensor_scatter_nd_update(fields, indices=[[i]], updates=field)

        return fields

    def _affaine_generator(self, batch_shape: ShapeBreakDown) -> tf.Tensor:
        matrices = batch_random_image_affine_matrices(
            batch_shape.core_shape_list,
            self._config.affine_origin_shift_range,
            self._config.affine_translations_range,
            self._config.affine_rotations_range,
            self._config.affine_shears_range,
            self._config.affine_scales_ranges,
            batch_shape.batch_size_int,
            inverse=True,  # flow fields are inverse of transformations
            random_generator=self._affine_random_generator,
        )

        return flow_fields_from_batch_affine_matrices(
            matrices,
            batch_shape.core_shape_list,
            self._config.float_data_type,
        )

    def _non_rigid_generator(self, batch_shape: ShapeBreakDown) -> tf.Tensor:
        return self._non_rigid_fields.generate_randomized(
            self._config.non_rigid_initial_seed_range,
            self._config.non_rigid_max_magnitude_range,
            batch_shape,
            self._nonrigid_random_generator,
        )

    def _combined_generator(self, batch_shape: ShapeBreakDown) -> tf.Tensor:
        return self._affaine_generator(batch_shape) + self._non_rigid_generator(
            batch_shape
        )

    def _truncated_white_noise(self, batch_shape: ShapeBreakDown) -> tf.Tensor:
        return truncated_white_noise(
            batch_shape,
            self._noise_random_generator,
            self._config.white_noise_flows_threshold,
            self._config.float_data_type,
        )
