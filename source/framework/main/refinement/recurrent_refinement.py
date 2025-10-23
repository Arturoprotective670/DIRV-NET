from typing import Any, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from source.framework.main.refinement.recurrent_refinement_data import (
    RecurrentRefinementData,
)
from source.framework.tools.general import (
    StatisticsData,
    calc_statistics,
    combine_statistics,
)
from source.model.main.dirv_net import DIRVNet
from source.model.main.pipe_line.pipe_line_data import PipeLineData


def per_batch_recurrent_refinement(
    fixed_images: tf.Tensor,
    moving_images: tf.Tensor,
    extra_loss_function_args: Optional[List[Any]],
    max_recurrent_refinement_iterations: int,
    model: DIRVNet,
) -> List[RecurrentRefinementData]:
    """
    Performs recurrent refinement, by feeding the resultant registered image
    into the network again, until we get least standard deviation in the
    difference between registered and fixed image.

    Notes
    -----
    - It applies the recurrent refinement on the whole batch, i.e. smaller
        std is measured on the whole batch.

    Returns
    -------
    - List of `RecurrentRefinementData` objects.
    """

    recurrent_moving_image = moving_images
    recurrent_ground_truth_displacements = extra_loss_function_args[0]

    best_stats = StatisticsData(np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)

    results: List[RecurrentRefinementData] = []

    for iteration_index in range(max_recurrent_refinement_iterations):
        pipeline_data, loss, extra = model.forward_propagate(
            fixed_images,
            recurrent_moving_image,
            False,
            [recurrent_ground_truth_displacements, extra_loss_function_args[1]],
        )

        registered_images = pipeline_data.final_variation_unit().registered_images

        stats = calc_statistics((fixed_images - registered_images) * 255)

        if stats.standard_deviation <= best_stats.standard_deviation:
            best_stats = stats
        else:
            break

        recurrent_moving_image = registered_images
        recurrent_ground_truth_displacements -= (
            pipeline_data.final_variation_unit().predicted_displacements
        )

        results.append(
            RecurrentRefinementData(
                pipeline_data, loss.numpy(), stats, extra, iteration_index
            )
        )

    return results


def per_image_recurrent_refinement(
    fixed_images: tf.Tensor,
    moving_images: tf.Tensor,
    extra_loss_function_args: Optional[List[Any]],
    max_recurrent_refinement_iterations: int,
    model: DIRVNet,
) -> Tuple[
    List[List[RecurrentRefinementData]],
    RecurrentRefinementData,
    RecurrentRefinementData,
]:
    """
    Breaks down the tensors of images' batch to individual images, and
    performs recurrent refinement on each image separately, and combines the
    results.

    Returns
    -------
    - List of `RecurrentRefinementData` List objects.
    - Combined `RecurrentRefinementData` object, of the first refinement
        results from all images.
    - Combined `RecurrentRefinementData` object, of the best refinement
        results (i.e. last result) from all images.
    """
    all_results: List[List[RecurrentRefinementData]] = []

    combined_first_result: Optional[RecurrentRefinementData] = None
    combined_best_result: Optional[RecurrentRefinementData] = None

    total_iterations_count = 0

    fixed_images_list = tf.unstack(fixed_images, axis=0)
    moving_images_list = tf.unstack(moving_images, axis=0)

    ground_truth_displacement = extra_loss_function_args[0]
    gt_displacement_list = tf.unstack(ground_truth_displacement, axis=0)

    for index, (moving_image, fixed_image, gt_displacement) in enumerate(
        zip(moving_images_list, fixed_images_list, gt_displacement_list)
    ):
        moving_image = tf.expand_dims(moving_image, axis=0)
        fixed_image = tf.expand_dims(fixed_image, axis=0)
        gt_displacement = tf.expand_dims(gt_displacement, axis=0)

        results = per_batch_recurrent_refinement(
            fixed_image,
            moving_image,
            [gt_displacement] + extra_loss_function_args[1:],
            max_recurrent_refinement_iterations,
            model,
        )

        total_iterations_count += len(results)

        if combined_first_result is None:
            combined_first_result = results[0].clone()
            combined_first_result.statistics = [combined_first_result.statistics]
        else:
            combined_first_result.pipeline_data = _combine_pipeline_data(
                results[0].pipeline_data, combined_first_result.pipeline_data
            )
            combined_first_result.statistics.append(results[0].statistics)
            # merge storage loss
            combined_first_result.extras_from_loss_function[0].merge_storage(
                results[0].extras_from_loss_function[0]
            )
        assert (
            combined_first_result.pipeline_data.final_variation_unit().registered_images.shape[
                0
            ]
            == index + 1
        )

        if combined_best_result is None:
            combined_best_result = results[-1].clone()
            combined_best_result.statistics = [combined_best_result.statistics]
        else:
            combined_best_result.pipeline_data = _combine_pipeline_data(
                results[-1].pipeline_data, combined_best_result.pipeline_data
            )
            combined_best_result.statistics.append(results[-1].statistics)
            # merge storage loss
            combined_best_result.extras_from_loss_function[0].merge_storage(
                results[0].extras_from_loss_function[0]
            )

        assert (
            combined_best_result.pipeline_data.final_variation_unit().registered_images.shape[
                0
            ]
            == index + 1
        )

        all_results.append(results)

    # TODO: there is a bug somewhere in how the results are combined

    # === setup the rest of combined first results info
    # average number of iterations
    combined_first_result.pseudo_iteration = 1
    # combined final std
    combined_first_result.statistics = combine_statistics(
        combined_first_result.statistics
    )
    # calculate actual combined loss
    approximate_gt_displacements = (
        ground_truth_displacement
        - combined_first_result.pipeline_data.final_variation_unit().predicted_displacements
    )
    extra = [approximate_gt_displacements] + extra_loss_function_args[1:]
    combined_first_result.pseudo_loss = (
        model.config.surrogate_loss_function_instance.loss(
            combined_first_result.pipeline_data, extra
        )[0].numpy()
    )

    assert (
        combined_first_result.pipeline_data.final_variation_unit().registered_images.shape[
            0
        ]
        == fixed_images.shape[0]
    )

    # === setup the rest of combined best results info
    # average number of iterations
    combined_best_result.pseudo_iteration = total_iterations_count / len(
        fixed_images_list
    )
    # combined final std
    combined_best_result.statistics = combine_statistics(
        combined_best_result.statistics
    )
    # calculate actual combined loss
    approximate_gt_displacements = (
        ground_truth_displacement
        - combined_best_result.pipeline_data.final_variation_unit().predicted_displacements
    )
    extra = [approximate_gt_displacements] + extra_loss_function_args[1:]
    combined_best_result.pseudo_loss = (
        model.config.surrogate_loss_function_instance.loss(
            combined_best_result.pipeline_data, extra
        )[0].numpy()
    )

    assert (
        combined_best_result.pipeline_data.final_variation_unit().registered_images.shape[
            0
        ]
        == fixed_images.shape[0]
    )

    return all_results, combined_first_result, combined_best_result


def _concat_tensors(
    new_tensor: Optional[tf.Tensor], base_tensor: Optional[tf.Tensor]
) -> Optional[tf.Tensor]:
    if new_tensor is None and base_tensor is not None:
        return tf.identity(base_tensor)
    elif base_tensor is None and new_tensor is not None:
        return tf.identity(new_tensor)
    elif new_tensor is None and base_tensor is None:
        return None

    return tf.concat([base_tensor, new_tensor], axis=0)


def _combine_pipeline_data(
    new_pipeline_data: PipeLineData, base_pipeline_data: PipeLineData
) -> PipeLineData:
    for new_pl_data, base_pl_data in zip(
        new_pipeline_data.pyramids, base_pipeline_data.pyramids
    ):
        for new_vu_data, base_vu_data in zip(
            new_pl_data.variational_units, base_pl_data.variational_units
        ):
            base_vu_data.registered_images = _concat_tensors(
                new_vu_data.registered_images, base_vu_data.registered_images
            )
            base_vu_data.predicted_displacements = _concat_tensors(
                new_vu_data.predicted_displacements,
                base_vu_data.predicted_displacements,
            )
            base_vu_data.predicted_flows = _concat_tensors(
                new_vu_data.predicted_flows, base_vu_data.predicted_flows
            )
            base_vu_data.ground_truth_displacements = _concat_tensors(
                new_vu_data.ground_truth_displacements,
                base_vu_data.ground_truth_displacements,
            )
            base_vu_data.ground_truth_flow = _concat_tensors(
                new_vu_data.ground_truth_flow, base_vu_data.ground_truth_flow
            )
            base_vu_data.input_displacements = _concat_tensors(
                new_vu_data.input_displacements, base_vu_data.input_displacements
            )
            base_vu_data.input_flows = _concat_tensors(
                new_vu_data.input_flows, base_vu_data.input_flows
            )
            base_vu_data.moving_images = _concat_tensors(
                new_vu_data.moving_images, base_vu_data.moving_images
            )
            base_vu_data.fixed_images = _concat_tensors(
                new_vu_data.fixed_images, base_vu_data.fixed_images
            )

    return base_pipeline_data
