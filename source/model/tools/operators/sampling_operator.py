"""
Purpose
-------
- Creates displacement field out of flow field, by sampling it at control points
  locations.

Contributors
------------
- TMS-Namespace
"""

from typing import Tuple, Union
import tensorflow as tf

from source.model.tools.shape_break_down import ShapeBreakDown


def sample(
    flow_field: tf.Tensor, spacings: Union[Tuple[int, int], Tuple[int, int, int]]
) -> tf.Tensor:
    rank = ShapeBreakDown(flow_field).core_rank_int

    if rank == 2:
        return flow_field[:, :: int(spacings[0]), :: int(spacings[1]), :]

    elif rank == 3:
        return flow_field[
            :, :: int(spacings[0]), :: int(spacings[1]), :: int(spacings[2]), :
        ]

    else:
        raise NotImplementedError
