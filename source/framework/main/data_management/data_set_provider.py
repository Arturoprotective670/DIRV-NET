from __future__ import annotations
from typing import List, Optional, Tuple, Union

import os
import numpy as np

from sklearn.model_selection import train_test_split

from source.framework.main.data_management.data_set import DataSet

from source.framework.settings.whole_config import WholeConfig
from source.framework.settings.enums import DataSets, Stages

from source.framework.tools.gradient_chessboard_generator import (
    random_gradient_chessboard,
)
from source.framework.tools.pather import Pather


class DataSetProvider:
    """
    Purpose
    --------
    - dataset loader, depending on the current stage, particular dataset will loaded.

    Notes
    -----
    - In this work, data had not a huge size, and even  3D image batches where
        fitting easily in RAM, so we did not need to implement lazy loading.

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(self, config: WholeConfig) -> None:
        self._config = config

        self._train_random_generator = np.random.default_rng(
            self._config.global_random_seed
        )
        self._test_random_generator = np.random.default_rng(
            self._config.test_global_random_seed
        )

    def load_data_set(
        self, stage: Stages
    ) -> Union[DataSet, Tuple[DataSet, Optional[DataSet]]]:
        data: List = []

        if self._config.data_set == DataSets.FASHION_MINST:
            data = self._load_fashion_mnist(stage)
        if self._config.data_set == DataSets.NRRD_SLICED:
            data = self._load_nrrd_data(stage)
        if self._config.data_set == DataSets.PATCHED_256_MONOCHROME_ANHIR:
            data = self._load_ANHIR_preprocessed(stage)
        if self._config.data_set == DataSets.SLICED_4DCT:
            data = self._load_sliced_4DCT(stage)
        if self._config.data_set == DataSets.INTERPOLATED_4DCT:
            data = self._load_interpolated_4DCT(stage)
        if self._config.data_set == DataSets.GRADIENT_CHESSBOARD_64:
            data = self._gradient_chessboard(64, (6, 10), stage)
        if self._config.data_set == DataSets.GRADIENT_CHESSBOARD_128:
            data = self._gradient_chessboard(128, (6, 10), stage)
        if self._config.data_set == DataSets.GRADIENT_CHESSBOARD_64_DENSE:
            data = self._gradient_chessboard(64, (3, 7), stage)

        if stage == Stages.TRAINING:
            return self._get_train_validation_datasets(data)
        elif stage == Stages.TESTING:
            return self._get_test_dataset(data)
        else:
            raise NotImplementedError

    def _load_fashion_mnist(self, stage: Stages) -> List:
        from keras.datasets import fashion_mnist

        (rest_of_data, _), (test_data, _) = fashion_mnist.load_data()

        if stage == Stages.TRAINING:
            return rest_of_data
        elif stage == Stages.TESTING:
            return test_data
        else:
            raise NotImplementedError

    def _load_nrrd_data(self, stage: Stages) -> List:
        DATA_DIR = "/data/nrrd_experiment/patches"
        IMAGES_EXTENSION = "*.jpg"

        if stage == Stages.TRAINING:
            return Pather(DATA_DIR).contents(IMAGES_EXTENSION)
        else:
            raise NotImplementedError

    def _load_ANHIR_preprocessed(self, stage: Stages) -> List:
        """
        Notes
        -----
        - You need to run
          `support/scripts/datasets_generation/generate_ANHIR_data_set.py`
          script to generate the data.
        - this should be normalized with `LOCAL_UNITY_NORMALIZATION`
        """
        TRAIN_DATA_DIR = "data/ANHIR_challenge/processed_265/train"
        TEST_DATA_DIR = "data/ANHIR_challenge/processed_265/test"
        IMAGES_EXTENSION = "*.jpg"

        assert os.path.isdir(TRAIN_DATA_DIR)

        if stage == Stages.TRAINING:
            return Pather(TRAIN_DATA_DIR).contents(IMAGES_EXTENSION)
        elif stage == Stages.TESTING:
            return Pather(TEST_DATA_DIR).contents(IMAGES_EXTENSION)
        else:
            raise NotImplementedError

    def _load_sliced_4DCT(self, stage: Stages) -> List:
        DIR_PATH = "data/4DCT_uncompressed/4DCT/"
        data: List[str] = []

        if stage == Stages.TRAINING:
            from support.scripts.file_formats.h5_format import read_h5

            for path in [Pather(DIR_PATH).contents("*.h5")[0]]:
                # for path in Pather(DIR_PATH).contents("*.h5"):
                res = read_h5(path)
                for i in range(res.shape[-1]):
                    r = res[..., i]  # [36:100, 36:100, 16: 80]
                    data.append(self._clip_4DCT(r))
                    # self.test_data_set.original_data.append(r)

            return data
        else:
            raise NotImplementedError

    def _gradient_chessboard(
        self,
        image_size: int,
        cells_size_range: Tuple[int, int],
        stage: Stages,
    ) -> None:
        """
        Notes
        -----
        - use `GLOBAL_UNITY_NORMALIZATION` with this.
        """
        if stage == Stages.TRAINING:
            return self._get_gradient_chessboard_list(
                self._config.crop_input_dataset_size_to,
                image_size,
                cells_size_range,
                self._train_random_generator,
            )
        elif stage == Stages.TESTING:
            return self._get_gradient_chessboard_list(
                self._config.crop_test_dataset_size_to,
                image_size,
                cells_size_range,
                self._test_random_generator,
            )
        else:
            raise NotImplementedError

    def _get_gradient_chessboard_list(
        self,
        images_count: int,
        image_size: int,
        random_cells_size_range: Tuple[int, int],
        random_generator: np.random.Generator,
    ) -> List:
        images_shape = [image_size] * self._config.images_rank

        results = []
        for _ in range(images_count):
            img = random_gradient_chessboard(
                images_shape,
                random_cells_size_range=random_cells_size_range,
                channels_count=1,
                random_generator=random_generator,
            )
            results.append(img)

        return results

    def _load_interpolated_4DCT(self, stage: Stages) -> List:
        DIR_PATH = "data/h5_my_interpolated_2x2x2"

        from support.scripts.file_formats.h5_format import read_h5

        self.validation_data_set.original_data = []

        data: List = []

        if stage == Stages.TRAINING:
            for path in Pather(DIR_PATH).deeper("train").contents("*.h5"):  # [5:8]:
                res = read_h5(path)
                res = self._clip_4DCT(res)
                data.append(res)
        elif stage == Stages.TRAINING:
            for path in Pather(DIR_PATH).deeper("test").contents("*.h5"):  # [0:1]:
                res = read_h5(path)
                res = self._clip_4DCT(res)
                data.append(res)
        else:
            raise NotImplementedError

        return data

    def _clip_4DCT(self, image):
        # this is according to Isotropic Total Variation Regularization of
        # Displacements in Parametric Image Registration
        # Valeriy Vishnevskiy, Christine Tanner, Orcun Goksel et. al. paper
        MAX_CLIP = 1200
        MIN_CLIP = 50

        image = np.clip(image, MIN_CLIP, MAX_CLIP)
        return (image - MIN_CLIP) / (MAX_CLIP - MIN_CLIP)

    def _get_train_validation_datasets(self, data) -> Tuple[DataSet, Optional[DataSet]]:
        if (
            self._config.crop_input_dataset_size_to is not None
            and len(data) > self._config.crop_input_dataset_size_to
        ):
            data = data[: self._config.crop_input_dataset_size_to]

        if self._config.train_to_validation_data_splitting_ratio != 0:
            train_data, validation_data = train_test_split(
                data,
                train_size=self._config.train_to_validation_data_splitting_ratio,
                random_state=self._config.global_random_seed,
                shuffle=self._config.shuffle_data_set,
            )
        else:
            train_data = data
            validation_data = []

        if len(train_data) > 0:
            train_ds = DataSet(
                self._config, self._train_random_generator, Stages.TRAINING
            )
            train_ds.original_data = train_data
        else:
            raise Exception

        validate_ds: Optional[DataSet] = None
        if len(validation_data) > 0:
            validate_ds = DataSet(
                self._config, self._train_random_generator, Stages.VALIDATION
            )
            validate_ds.original_data = validation_data

        return train_ds, validate_ds

    def _get_test_dataset(self, test_data) -> DataSet:
        if (
            self._config.crop_test_dataset_size_to is not None
            and len(test_data) > self._config.crop_test_dataset_size_to
        ):
            test_data = test_data[: self._config.crop_test_dataset_size_to]

        if len(test_data) > 0:
            ds = DataSet(self._config, self._test_random_generator, Stages.TESTING)
            ds.original_data = test_data
            return ds
        else:
            raise Exception
