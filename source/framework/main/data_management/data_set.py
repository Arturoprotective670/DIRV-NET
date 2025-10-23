from __future__ import annotations
from numpy.typing import NDArray
from typing import List, Optional, Tuple, Union

import math
import numpy as np
from PIL import Image

import tensorflow as tf

from source.framework.settings.whole_config import WholeConfig
from source.framework.settings.enums import Stages

from source.framework.main.data_management.batch import Batch
from source.framework.main.data_management.losses_storage import LossesStorage

from source.framework.tools.general import StatisticsData
from source.framework.tools.pather import Pather
from source.framework.tools.patches_generator import random_array_patches

from source.framework.settings.synthetic_fields_provider import SyntheticFieldsProvider


class DataSet:
    """
    Purpose
    --------
    - hold various data information about data set, this
    includes displacements, flow fields, as well as losses.
    Additionally, it will be responsible for image patch generating.

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(
        self, config: WholeConfig, random_generator: np.random.Generator, stage: Stages
    ) -> None:
        self._config = config
        self.stage = stage
        self._random_generator = random_generator

        # original data may hold image arrays, as well as image file paths,
        # that after reading, will put in fixed images
        self.original_data: List[Union[NDArray, np.str_]] = None

        # prediction losses
        self.losses_storage: LossesStorage = LossesStorage()
        self.initial_losses_storage: LossesStorage = LossesStorage()

        # the frozen batch will be the first batch, that will be returned as the
        # last batch in each epoch, this will guarantee that it will not depend
        # on the the number of patches, or number of images, or images count in
        # the last batch, same for the synthetic fields, since random generating
        # of those will change, and only the global random seed will has an
        # effect.
        self._frozen_batch: Optional[Batch] = None
        # we will also need a the value of the data cursor of the first frozen
        # batch, to start from in the following epoch.
        self._data_cursor: int = 0
        self._frozen_data_cursor: int = 0

        self.current_batch: Optional[Batch] = None

        # we need two of below, because the internal one indicates that we went
        # over whole data set, and the outer one will indicate the actual final
        # batch, since we may need also to return the  frozen batch, before we
        # really finish.
        self.is_final_batch: bool = False
        self._is_final_batch: bool = False

        self._synthetic_fields = SyntheticFieldsProvider(config, stage)

        self._reset_batches_iterator()

        if stage == Stages.TRAINING:
            self._default_batch_size = config.training_batch_size
        elif stage == Stages.VALIDATION:
            self._default_batch_size = config.validation_default_batch_size
        elif stage == Stages.TESTING:
            self._default_batch_size = config.testing_default_batch_size
        else:
            raise NotImplementedError

        self.ground_truth_flows_magnitude_statistics :List[StatisticsData] = []
        self.ground_truth_flows_component_statistics :List[StatisticsData] = []

    # region === Help functions

    def _reset_batches_iterator(self) -> None:
        self._patches_list_queue = []
        # data_cursor counts how many images had been generated so far, this may
        # be different from batched_images_count, in case of patch generation.
        # we set it to the frozen data cursor, to start from there in the next
        # epoch (it will be always zero if freezing not requested)
        self._data_cursor = self._frozen_data_cursor
        # the number of images that been already passed to batched (this may be
        # different of dataset size due for patch generating for example)
        self.batched_images_count = 0
        self.current_batch_index = 0
        self._skipped_images_count = 0

        self.current_batch = None

        self.is_final_batch = False
        self._is_final_batch = False

    def _next_batch(self) -> Batch:
        # get batch data as a list
        batch_data_list = self._next_batch_raw_data_list()
        # convert to tensor
        batch_data = self._data_list_to_tensor(batch_data_list)
        # setup batch object
        batch = Batch(self._config, self._synthetic_fields)
        batch.original_images = batch_data
        # increase the number of batched items
        self.batched_images_count += len(batch_data)
        # initialize data
        batch.initialize_batch_data()

        self.current_batch = batch

        return batch

    def _next_batch_raw_data_list(self) -> List:
        data_list = None
        data_set_length = self.size()

        # patching is applicable only during training
        if self._config.training_patches_count > 0 and self.stage == Stages.TRAINING:
            patching_set_size = data_set_length
            patching_set_size *= self._config.training_patches_count

            batch_size = min(
                patching_set_size - self._data_cursor, self._default_batch_size
            )

            data_list = []
            for _ in range(batch_size):
                # choose a random image, we want the batch to consist of patches
                # of random images, and not of one image patches
                image = self.original_data[
                    self._random_generator.choice(range(data_set_length))
                ]
                image = self._read_data([image])[0]

                # generate a random patches of the image
                patches = []
                while len(patches) == 0:  # in case threshold will discard all images
                    patches = self._get_patches(image, 1)
                data_list.append(patches[0])

                self._data_cursor += 1

            self._is_final_batch = self._data_cursor >= patching_set_size

        else:
            if self._data_cursor <= data_set_length - 1:
                batch_size = min(
                    data_set_length - self._data_cursor, self._default_batch_size
                )

                data_list = self.original_data[
                    self._data_cursor : self._data_cursor + batch_size
                ]
                data_list = self._read_data(data_list)

                self._data_cursor += batch_size

            self._is_final_batch = self._data_cursor >= data_set_length

        assert len(data_list) > 0

        return data_list

    def _get_patches(self, image: NDArray, count: int) -> List[NDArray]:
        patches = random_array_patches(
            image,
            self._config.training_patches_shape,
            count,
            random_generator=self._random_generator,
        )
        valid_patches = []
        if self._config.empty_patch_max_std_threshold > 0:
            for patch in patches:
                if np.std(patch) > self._config.empty_patch_max_std_threshold:
                    # mark image corner, to see if any accident flipping takes place
                    # patch[0:2, 0:2, :] = 1
                    # patch[2:4, 0:2, :] = 0
                    # patch[0:2, 2:4, :] = 0
                    # patch[2:4, 2:4, :] = 1

                    # save patch to file
                    # img = patch
                    # img = img[..., img.shape[-1] // 2 ] * 255
                    # img = Image.fromarray(np.uint8(img)).convert("L")
                    # img.save(Pather("patches").join(str(datetime.now()) + ".jpg"))

                    valid_patches.append(patch)
                # else:
                #     # save discarded patches
                #     img = patch
                #     img = img[..., img.shape[-1] // 2 ] * 255
                #     img = Image.fromarray(np.uint8(img)).convert("L")
                #     img.save(Pather("discarded_patches").join(str(datetime.now()) + ".jpg"))
            # count the number of the skipped patches
            diff = len(patches) - len(valid_patches)
            if diff > 0:
                self._skipped_images_count += diff
                # print(f"A {diff}/{self._config.training_patches_count} almost empty patches are skipped.")

        else:
            valid_patches = patches

        return valid_patches

    def _read_data(self, data: List) -> List:
        """
        A help function, that will check if the data is provided as file paths,
        it reads them.
        """
        if isinstance(data[0], str):
            pather = Pather(data[0])

            if pather.extension in [".jpg"]:
                return [np.array(Image.open(image_path)) for image_path in data]
                # return self._data_list_to_tensor(data)

            # elif pather.extension == ".npy":
            #     # TODO: not fully implemented
            #     self.load_data(data)
            #     return data
            else:
                raise Exception(f"File type ({pather.extension}) is not supported.")
        else:
            return data

    def _data_list_to_tensor(self, data: List) -> tf.Tensor:
        """
        A help function, that converts a `list` to `Tensor`, and will check if
        channel info is provided or not, if not, it will add a dummy dimension
        to the tensor, for tensorflow compatibility.
        """

        data = tf.convert_to_tensor(data, dtype=self._config.float_data_type)
        data_rank = tf.rank(data).numpy()

        if self._config.images_rank == 2:
            if data_rank == 3:
                # if the declared channels count exceeds one, but data dose not
                # contain any channels info, then something is wrong
                if self._config.channels_count > 1:
                    raise Exception("Wrong channels count.")
                else:
                    return tf.expand_dims(data, axis=-1)
            elif data_rank == 4:  # channels are already in data
                return data
            else:
                raise Exception("Wrong data rank")

        elif self._config.images_rank == 3:
            if data_rank == 4:
                # if the declared channels count exceeds one, but data dose not
                # contain any channels info, then something is wrong
                if self._config.channels_count > 1:
                    raise Exception("Wrong channels count.")
                else:
                    return tf.expand_dims(data, axis=-1)
            elif data_rank == 5:  # channels are already in data
                return data
            else:
                raise Exception("Wrong data rank")

        else:
            raise NotImplementedError

    # endregion

    # region === Public functions
    def size(self) -> int:
        assert self.original_data is not None
        return len(self.original_data)

    def images_shape(self) -> Union[Tuple[int, int], Tuple[int, int, int]]:
        first_image = self._read_data(self.original_data[0:1])
        return first_image[0].shape

    def append_batch_data(self, batch: Batch) -> None:
        """
        Mergers losses from another data object, to the current one, after
        converting them to `numpy`.
        Useful to merge the batch processing results, to the whole/full data set
        results.
        """
        self.losses_storage.merge_storage(batch.losses_storage)

        self.initial_losses_storage.merge_storage(
            self._config.surrogate_loss_function_instance.batch_initial_losses(batch)
        )

        self.ground_truth_flows_component_statistics.append(batch.ground_truth_flows_components_statistics())
        self.ground_truth_flows_magnitude_statistics.append(batch.ground_truth_flows_magnitude_statistics())

    def estimate_batches_count(self) -> int:
        estimated_images_to_batch = len(self.original_data)

        if self._config.training_patches_count > 0 and self.stage == Stages.TRAINING:
            estimated_images_to_batch *= self._config.training_patches_count
            # estimated_images_to_batch -= self._skipped_images_count

        return math.ceil(estimated_images_to_batch / self._default_batch_size)

    def batches_iterator(self) -> np.Generator[Batch]:
        # Using tensorflow Dataset is been showed useless in this project since
        # its main usefulness will be to load data in parallel and lazy way, to
        # avoid data loading bottleneck, however because we are performing
        # complex transformations on the batch, it fails to parallelize the
        # DataLoader, for now no solution is found.
        while not self._is_final_batch:
            # if we should freeze, and if this is the first batch of first epoch
            # (so we have no frozen batch yet), make the first batch a frozen
            # one, and return the second one.
            if (
                self._config.freeze_last_batch
                and self.current_batch_index == 0
                and self._frozen_batch is None
            ):
                self._frozen_batch = self._next_batch()
                self._frozen_data_cursor = self._data_cursor
                # we may have only one batch available in principle, then just
                # return the frozen one
                if self._is_final_batch:
                    self.is_final_batch = True
                    self.current_batch_index += 1
                    self.current_batch = self._frozen_batch
                    yield self._frozen_batch
                else:
                    self.current_batch_index += 1
                    yield self._next_batch()
            else:
                self.current_batch_index += 1
                # if we have only one batch and not in first epoch (so we have a
                # frozen one), then we return frozen one
                if (
                    self.estimate_batches_count() == 1
                    and self._frozen_batch is not None
                ):
                    self._is_final_batch = True
                    self.is_final_batch = True
                    self.current_batch = self._frozen_batch
                    yield self._frozen_batch

                yield self._next_batch()

        # we went over the whole dataset, so return the first frozen batch
        if self._is_final_batch and not self.is_final_batch:
            self.is_final_batch = True
            self.current_batch_index += 1
            self.current_batch = self._frozen_batch
            yield self._frozen_batch

        # we fully done, reset the iterator, and raise StopIteration exception
        # (done automatically when no yield)
        if self.is_final_batch:
            self._reset_batches_iterator()

    def save_losses(self) -> None:
        self.losses_storage.save(self.stage.name.lower(), self._config)

    # def load_data(self, file_paths : List[str]):
    #     fixed_data = []
    #     moving_data = []
    #     registered_data = []

    #     ground_truth_transformations = []
    #     ground_truth_flow_fields = []

    #     ground_truth_inverse_flow_fields = []
    #     ground_truth_inverse_transformations = []

    #     for file_path in file_paths:
    #         with open(file_path, 'rb') as file_handle:
    #             fixed_data.append(np.load(file_handle))
    #             moving_data.append(np.load(file_handle))
    #             registered_data.append(np.load(file_handle))
    #             ground_truth_transformations.append(np.load(file_handle))
    #             ground_truth_flow_fields.append(np.load(file_handle))
    #             ground_truth_inverse_flow_fields.append(np.load(file_handle))
    #             ground_truth_inverse_transformations.append(np.load(file_handle))

    #     dtype = self._config.float_data_type

    #     # TODO: add to variables, not replace them
    #     self.fixed_data = \
    #         tf.cast(fixed_data, dtype = dtype)
    #     self.moving_data = \
    #         tf.cast(moving_data, dtype = dtype)
    #     self.registered_data = \
    #         tf.cast(registered_data, dtype = dtype)
    #     self.ground_truth_transformations = \
    #         tf.cast(ground_truth_transformations, dtype = dtype)
    #     self.ground_truth_flow_fields = \
    #         tf.cast(ground_truth_flow_fields, dtype = dtype)
    #     self.ground_truth_inverse_flow_fields = \
    #         tf.cast(ground_truth_inverse_flow_fields, dtype = dtype)
    #     self.ground_truth_inverse_transformations = \
    #         tf.cast(ground_truth_inverse_transformations, dtype = dtype)

    # endregion
