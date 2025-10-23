from typing import List, Optional, Tuple
from types import FunctionType

import time
import datetime

import numpy as np
import tensorflow as tf

import os

from pprint import pprint
from inspect import getmembers


from source.framework.main.refinement.recurrent_refinement_data import (
    RecurrentRefinementData,
)
from source.framework.tools.gpu_tools import get_gpu_info
from source.model.main.dirv_net import DIRVNet
from source.framework.settings.whole_config import WholeConfig
from source.framework.tools.general import (
    StatisticsData,
    block_mean,
    calc_statistics,
    combine_statistics,
)
from source.framework.main.data_management.data_set import DataSet, Stages


class LogReporter:
    """
    Purpose
    -------
    - Various functions to generate log files.

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(self, config: WholeConfig) -> None:
        self._config: WholeConfig = config

        self._logger, self._logclose = self._create_logger(config.log_path)

        self._train_flows_statistics: List[StatisticsData] = []
        self._train_flows_magnitude_statistics: List[StatisticsData] = []

        self._test_flows_statistics: List[StatisticsData] = []
        self._test_flows_magnitude_statistics: List[StatisticsData] = []

        self._previous_epoch_stats: Tuple[
            Optional[StatisticsData], Optional[StatisticsData], Optional[StatisticsData]
        ] = [None, None, None]

        self.do_not_log: bool = False

    # region === Help Functions

    def _attributes(self, obj):
        disallowed_names = {
            name
            for name, value in getmembers(type(obj))
            if isinstance(value, FunctionType)
        }
        return {
            name: getattr(obj, name)
            for name in dir(obj)
            if name[0] != "_" and name not in disallowed_names and hasattr(obj, name)
        }

    def _print_attributes(self, obj) -> None:
        pprint(self._attributes(obj))

    def _create_logger(self, log_filename: str, display=True):
        # TODO: rewrite this
        f = open(log_filename, "a")
        counter = [0]

        # this function will still have access to f after create_logger
        # terminates
        def logger(text):
            if display:
                print(text)
            f.write(text + "\n")
            counter[0] += 1
            if counter[0] % self._config.flash_text_logs_every_lines == 0:
                f.flush()
                os.fsync(f.fileno())
            # Question: do we need to flush()

        return logger, f.close

    def _get_loss_block_mean(
        self, initial_losses, losses, mean_block_size: int, stage: Stages
    ) -> List:
        if stage == Stages.TRAINING:
            losses = block_mean(losses, mean_block_size)
        # insert the initial losses at the beginning we do not want initial
        # values to be averaged
        return [np.mean(initial_losses)] + losses

    def _get_time_difference(self, start_time: time, multiplier: float = 1) -> str:
        return str(
            datetime.timedelta(seconds=round((time.time() - start_time) * multiplier))
        )

    def _get_improvement_of_means(
        self,
        new_stats: StatisticsData,
        old_stats: Optional[StatisticsData],
        invert: bool,
    ) -> float:
        if old_stats is None:
            return 0.0

        if invert:
            return 100 * (old_stats.mean - new_stats.mean) / new_stats.mean
        else:
            return 100 * (new_stats.mean - old_stats.mean) / old_stats.mean

    # endregion

    # region === Public Functions

    def report_losses_stats(
        self, stage: Stages, epoch_index: int, losses_list: List
    ) -> None:
        stage_name = stage.name.capitalize() + " "

        # initial metrics, will have -1 index
        INITIAL_PREFIX = "Initial "
        if epoch_index == -1:
            stage_name += INITIAL_PREFIX

        if epoch_index == 0:
            self._previous_epoch_stats = [None, None, None]

        losses_stats = calc_statistics(losses_list[0])

        intensities_stats = calc_statistics(losses_list[1])

        displacements_stats = calc_statistics(losses_list[2])

        # when we getting the initial metrics (epoch = -1), things are reversed
        # in improvement calculations
        self.log_line(
            (
                "\n{0}intensity dissimilarity:\t\t{1:07.4f} ± {2:07.4f} ({3:05.2f}%)\timprovement: {4:05.3f}%\n"
                + "{0}displacements dissimilarity:\t{5:07.4f} ± {6:07.4f} ({7:05.2f}%)\timprovement: {8:05.3f}%\n"
                + "{0}loss:\t\t\t\t\t\t{9:07.4f} ± {10:07.4f} ({11:05.2f}%)\timprovement: {12:05.3f}%\n"
            ).format(
                "----  " + stage_name,
                intensities_stats.mean,
                intensities_stats.standard_deviation,
                intensities_stats.coefficient_of_variation,
                self._get_improvement_of_means(
                    intensities_stats, self._previous_epoch_stats[0], epoch_index == -1
                ),
                displacements_stats.mean,
                displacements_stats.standard_deviation,
                displacements_stats.coefficient_of_variation,
                self._get_improvement_of_means(
                    displacements_stats,
                    self._previous_epoch_stats[1],
                    epoch_index == -1,
                ),
                losses_stats.mean,
                losses_stats.standard_deviation,
                losses_stats.coefficient_of_variation,
                self._get_improvement_of_means(
                    losses_stats, self._previous_epoch_stats[2], epoch_index == -1
                ),
            )
        )

        self._previous_epoch_stats = [
            intensities_stats,
            displacements_stats,
            losses_stats,
        ]

    def report_epoch_start(
        self, epoch_index, current_temperature, learning_rate
    ) -> None:
        self.log_line(
            (
                "=======\t"
                + "staring epoch: {0:03.0f}/{1:03.0f}\t"
                + "temperature: {2:08.6f}\t"
                + "learning rate: {3:02.6f}\n"
            ).format(
                epoch_index + 1,
                self._config.training_epochs_count,
                current_temperature,
                learning_rate,
            )
        )

    def report_epoch_end(self, total_time: time, epoch_index: int) -> None:
        self.log_line(
            "estimated total time:\t{}/{}\n".format(
                self._get_time_difference(total_time),
                self._get_time_difference(
                    total_time,
                    (self._config.training_epochs_count - self._config.start_from_epoch)
                    / (epoch_index + 1),
                ),
            )
        )

    def report_batch_progress(
        self,
        data_set: DataSet,
        batches_staring_time: time,
    ) -> None:
        # we report the mean of the loses since last report
        loses_to_count = self._config.report_batches_progress_every

        if data_set.is_final_batch:  # the final batch, could be of a different size
            modulo = (
                data_set.current_batch_index
                % self._config.report_batches_progress_every
            )
            if modulo != 0:
                loses_to_count = modulo

        batches_losses = data_set.losses_storage.per_image_combined_loss[
            -loses_to_count:
        ]

        if data_set.current_batch_index == 0:
            self.log_line(
                (
                    "processed: 000/{0:03.0f} batches\t"
                    + "time: 0:00:00/0:00:00\t"
                    + "batches loss: 00.0000\t"
                    + "flow max: 00.00"
                ).format(data_set.estimate_batches_count())
            )
        else:
            stats = data_set.current_batch.ground_truth_flows_magnitude_statistics()

            multiplier = (
                data_set.estimate_batches_count() / data_set.current_batch_index
            )

            self.log_line(
                (
                    "processed: {0:03.0f}/{1:03.0f} batches\t"
                    + "time: {2}/{3}\t"
                    + "batches loss: {4:07.4f}\t"
                    + "flow max: {5:05.2f}"
                ).format(
                    data_set.current_batch_index,
                    data_set.estimate_batches_count(),
                    self._get_time_difference(batches_staring_time),
                    self._get_time_difference(batches_staring_time, multiplier),
                    np.mean(batches_losses),
                    stats.maximum,
                )
            )

    def report_batch_loop_end(self, epoch_index: int, data_set: DataSet) -> None:
        # data_set will contain losses for multiple epochs, so we need
        # to select only the last epoch.
        # batches loop iterator, resets batched_images_count every epoch
        images_count = data_set.batched_images_count

        losses_list = [
            data_set.losses_storage.per_image_combined_loss[-images_count:],
            data_set.losses_storage.per_image_intensity_losses[-images_count:],
            data_set.losses_storage.per_image_displacements_losses[-images_count:],
        ]

        self.log_line(
            "\npeak used GPU memory:\t{:04.3f} GB.".format(
                int(tf.config.experimental.get_memory_info("GPU:0")["peak"]) / (1024**3)
            )
        )

        self.report_losses_stats(data_set.stage, epoch_index, losses_list)

    def report_stage_start(self, stage: Stages) -> None:
        self.log_line(
            "\n////////\tStarting {} ...\n".format(str(stage.name).capitalize())
        )

    def report_stage_end(self, data_sets: List[DataSet], starting_time: time) -> None:
        """
        Prints an end of stage report, which include reporting the initial
        metrics that was before starting training or testing, and adds those
        values as a first values in tensorboard. Finally, reports elapsed time.
        """
        for data_set in data_sets:
            losses_list = [
                data_set.initial_losses_storage.per_image_combined_loss,
                data_set.initial_losses_storage.per_image_intensity_losses,
                data_set.initial_losses_storage.per_image_displacements_losses,
            ]

            stage_name = data_set.stage.name.capitalize() + " "

            self.report_losses_stats(data_set.stage, -1, losses_list)

        # show flows statistics
        stats = combine_statistics(data_set.ground_truth_flows_magnitude_statistics)

        self.log_line(
            (
                "\nFlow vectors magnitude statistics:\t"
                + "max: {0:05.2f}\t"
                + "min: {1:05.2f}\t"
                + "mean: {2:05.2f} ± {3:05.2f} ({4:05.2f}%)"
            ).format(
                stats.maximum,
                stats.minimum,
                stats.mean,
                stats.standard_deviation,
                stats.coefficient_of_variation,
            )
        )

        stats = combine_statistics(data_set.ground_truth_flows_component_statistics)

        self.log_line(
            (
                "Flow vector components statistics:\t"
                + "max: {0:05.2f}\t"
                + "min {1:05.2f}\t"
                + "mean: {2:05.2f} ± {3:05.02f} ({4:05.2f}%)\n"
            ).format(
                stats.maximum,
                stats.minimum,
                stats.mean,
                stats.standard_deviation,
                stats.coefficient_of_variation,
            )
        )

        self.report_elapsed_time(starting_time, stage_name + "total time")
        self.log_line("\n")

    def report_data_set(self, data_set: DataSet, name: str) -> None:
        if data_set is not None:
            self.log_line(
                f"{name} images count:\t{str(data_set.size())}\nimages shape:\t{str(data_set.images_shape())}\n"
            )
        else:
            self.log_line(f"No {name} data is provided.\n")

    def report_elapsed_time(self, from_time: time, text: str, suffix: str = "") -> None:
        self.log_line(f"{text}:\t\t{self._get_time_difference(from_time)}{suffix}")

    def log_line(self, text: str) -> None:
        if not self.do_not_log:
            self._logger(text)

    def report_model(self, model: DIRVNet) -> None:
        self.log_line("\n++++++ model parameters :")

        [
            self.log_line(
                f"{str(v._shared_name)}\t\t{str(v.shape)} = {np.prod(v.shape):,}"
            )
            for v in model.vars.to_list()
        ]

        params_count = np.sum([np.prod(v.shape) for v in model.vars.to_list()])

        self.log_line(f"\ntotal model parameters count:\t\t{params_count:,}")
        # mem = params_count * \
        #         np.prod(images_shape) * \
        #         self._config.batch_size* \
        #         self._config.channels_count* \
        #         0.15 / (1024**3) # multiplier is just an empirical coefficient
        # self.log_line("estimated training peak GPU memory:\t{:04.3f} GB".format(mem))
        self.log_line("++++++")

    def report_gpu(self) -> None:
        gpu_index = int(tf.config.get_visible_devices("GPU")[0].name[-1])

        device_count, _, gpu_name, total_memory = get_gpu_info(gpu_index)

        self.log_line(
            "\nThe used GPU (out of {0} GPUs): {1}\tAvailable GPU Memory: {2:.2f} Gb.\n".format(
                device_count, gpu_name, total_memory
            )
        )

    def report_refinement(
        self, first_data: RecurrentRefinementData, last_data: RecurrentRefinementData
    ) -> None:
        first = first_data.statistics.dissimilarity
        last = last_data.statistics.dissimilarity

        improvement = (first - last) * 100 / first

        self.log_line(
            f"recurrent refinement took {last_data.pseudo_iteration:5.2f} iterations,"
            + f" and improved dissimilarity of intensity diff. by {first:06.2f} -> {last:06.2f} ({improvement:05.2f}%)"
        )

    # endregion
