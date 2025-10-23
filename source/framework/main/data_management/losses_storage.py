import copy
from typing import List
import tensorflow as tf
import numpy as np

from source.framework.settings.whole_config import WholeConfig
from source.framework.tools.pather import Pather


class LossesStorage:
    """
    Purpose
    -------
    This class aims to save on disk, and store in memory, and merge the losses
    from various batches.

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(self) -> None:
        self.per_image_per_level_displacements_losses: List[float] = []
        self.per_image_per_level_intensity_losses: List[float] = []

        self.per_image_displacements_losses: List[float] = []
        self.per_image_intensity_losses: List[float] = []

        self.per_batch_per_level_displacements_losses: List[float] = []
        self.per_batch_per_level_image_intensity_losses: List[float] = []

        self.per_batch_displacements_losses: List[float] = []
        self.per_batch_intensity_losses: List[float] = []

        self.per_image_combined_loss: List[float] = []

    def merge_losses(
        self,
        combined_loss: tf.Tensor,
        displacements_losses: List[tf.Tensor],
        intensity_losses: List[tf.Tensor],
    ) -> None:
        self.per_image_combined_loss += list(combined_loss.numpy())

        displacements_losses = np.array(displacements_losses)
        intensity_losses = np.array(intensity_losses)

        self.per_image_per_level_displacements_losses += list(
            np.transpose(displacements_losses)
        )
        self.per_image_per_level_intensity_losses += list(
            np.transpose(intensity_losses)
        )

        self.per_image_displacements_losses += list(
            np.mean(displacements_losses, axis=0)
        )
        self.per_image_intensity_losses += list(np.mean(intensity_losses, axis=0))

        self.per_batch_per_level_displacements_losses += [
            list(np.mean(displacements_losses, axis=1))
        ]
        self.per_batch_per_level_image_intensity_losses += [
            list(np.mean(intensity_losses, axis=1))
        ]

        self.per_batch_displacements_losses += [np.mean(displacements_losses)]
        self.per_batch_intensity_losses += [np.mean(intensity_losses)]

    def merge_storage(self, loss_storage) -> None:
        self.per_image_combined_loss += loss_storage.per_image_combined_loss

        self.per_image_per_level_displacements_losses += (
            loss_storage.per_image_per_level_displacements_losses
        )
        self.per_image_per_level_intensity_losses += (
            loss_storage.per_image_per_level_intensity_losses
        )

        self.per_image_displacements_losses += (
            loss_storage.per_image_displacements_losses
        )
        self.per_image_intensity_losses += loss_storage.per_image_intensity_losses

        self.per_batch_per_level_displacements_losses += (
            loss_storage.per_batch_per_level_displacements_losses
        )
        self.per_batch_per_level_image_intensity_losses += (
            loss_storage.per_batch_per_level_image_intensity_losses
        )

        self.per_batch_displacements_losses += (
            loss_storage.per_batch_displacements_losses
        )
        self.per_batch_intensity_losses += loss_storage.per_batch_intensity_losses

    def save(self, stage_name: str, config: WholeConfig) -> None:
        """
        Saves the current losses into `csv` files.
        """

        directory_path = config.losses_saving_path
        directory = Pather(directory_path, default_extension="csv").deeper(stage_name)

        self._save_list_csv(
            self.per_image_displacements_losses,
            directory.join("displacements_dissimilarities"),
        )
        self._save_list_csv(
            self.per_image_intensity_losses, directory.join("intensity_dissimilarities")
        )

    def _save_list_csv(self, list: List[float], file_path: str) -> None:
        import csv

        with open(file_path, "w") as csv_file:
            write = csv.writer(csv_file)
            write.writerows([[item] for item in list])

    def clone(self) -> "LossesStorage":
        cp = copy.copy(self)
        return cp
