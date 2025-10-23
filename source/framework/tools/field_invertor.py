"""
Purpose
--------
- Calcs inversion of a flow fields.

Contributors
------------
- TMS-Namespace
- Claudio Fanconi
"""

from typing import Tuple
import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from source.model.tools.operators.sampling_operator import sample


def get_field_inverse(field: np.array, quality: int) -> np.array:
    """
    Inverts a vector field by using `SimpleITK` library.

    Arguments
    ---------
    - `field`: the array of the original field.
    - `quality`: the sampling quality, the lower the higher calculation quality
    is, but it highly affect the used memory and processing quality.

    Returns
    -------
    - The array of the inverted field.
    """

    image_array = sitk.GetImageFromArray(field, isVector=True)
    inverse_field_image = sitk.InverseDisplacementField(
        image_array, size=field.shape, subsamplingFactor=quality
    )
    return sitk.GetArrayFromImage(inverse_field_image)


def batch_flows_inverse(fields_batch: np.array, quality: int):
    """
    Inverts a batch of vector fields.

    Arguments
    ---------
    - `fields_batch`: a tensor of the original fields batch.
    - `quality`: the sampling quality, the lower the higher calculation quality
    is, but it highly affect the used memory and processing quality.

    Returns
    -------
    - A tensor of the inverted fields batch.
    """

    # invert each batch element separately
    inverted_fields_batch = [
        get_field_inverse(fields_batch[i], quality)
        for i in range(fields_batch.shape[0])
    ]
    # convert to tensor
    return tf.cast(inverted_fields_batch, tf.float32)


def inverted_fields_from_flows(
    flows: tf.Tensor, displacements_spacings: Tuple[int, int], quality: int
):
    """
    Inverts a batch of flow fields, and converts them to displacements fields
    (i.e. according to the displacements control points).

    Arguments
    ---------
    - `flows`: a tensor of the original flow fields batch.
    - `displacements_spacings`: the spacings between the control points of the
    displacement fields that we need to obtain.
    - `quality`: the sampling quality, the lower the higher calculation quality
    is, but it highly affect the used memory and processing quality.

    Returns
    -------
    - A tensor of the inverted displacements batch.
    - A tensor of the inverted flow fields batch.
    """
    # we are predicting the inverse fields ...
    flows_batch = batch_flows_inverse(flows, quality)

    # select the inverse fields at the displacements's control points
    # locations, since we are predicting the field at those points only
    displacements_batch = sample(flows_batch, displacements_spacings)

    return displacements_batch, flows_batch
