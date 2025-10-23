"""
Purpose
--------
- Contains classes to hold various pipeline data for easy of management.

Contributors
------------
- TMS-Namespace
"""

from typing import List, Optional

import tensorflow as tf

def _clone_tensor(tensor: Optional[tf.Tensor]) -> Optional[tf.Tensor]:
    if tensor is None:
        return None
    else:
        return tf.identity(tensor)

class VariationalUnitData:
    def __init__(self) -> None:
        self.input_displacements: Optional[tf.Tensor] = None
        self.input_flows: Optional[tf.Tensor] = None

        self.predicted_displacements: Optional[tf.Tensor] = None
        self.predicted_flows: Optional[tf.Tensor] = None

        self.ground_truth_displacements: Optional[tf.Tensor] = None
        self.ground_truth_flow: Optional[tf.Tensor] = None

        self.fixed_images: Optional[tf.Tensor] = None
        self.moving_images: Optional[tf.Tensor] = None

        self.registered_images: Optional[tf.Tensor] = None

    def clone(self) -> "VariationalUnitData":
        vu = VariationalUnitData()

        vu.input_displacements = _clone_tensor(self.input_displacements)
        vu.input_flows = _clone_tensor(self.input_flows)

        vu.predicted_displacements = _clone_tensor(self.predicted_displacements)
        vu.predicted_flows = _clone_tensor(self.predicted_flows)

        vu.ground_truth_displacements = _clone_tensor(self.ground_truth_displacements)
        vu.ground_truth_flow = _clone_tensor(self.ground_truth_flow)

        vu.fixed_images = _clone_tensor(self.fixed_images)
        vu.moving_images = _clone_tensor(self.moving_images)

        vu.registered_images = _clone_tensor(self.registered_images)

        return vu


class PyramidLevelData:
    def __init__(self) -> None:
        self.variational_units: List[VariationalUnitData] = []

    def last_variation_unit(self) -> Optional[VariationalUnitData]:
        if len(self.variational_units) == 0:
            return None
        else:
            return self.variational_units[-1]

    def registered_images(self) -> List[tf.Tensor]:
        return [vu_data.registered_images for vu_data in self.variational_units]

    def predicted_displacements(self) -> List[tf.Tensor]:
        return [vu_data.predicted_displacements for vu_data in self.variational_units]

    def clone(self) -> "PyramidLevelData":
        pl = PyramidLevelData()

        for vu in self.variational_units:
            pl.variational_units.append(vu.clone())

        return pl


class PipeLineData:
    def __init__(self) -> None:
        self.pyramids: List[PyramidLevelData] = []

    def last_pyramid(self) -> Optional[PyramidLevelData]:
        if len(self.pyramids) == 0:
            return None
        else:
            return self.pyramids[-1]

    def last_variation_unit(self) -> VariationalUnitData:
        return self.last_pyramid().last_variation_unit()

    def last_variation_units(self) -> List[VariationalUnitData]:
        vus = []

        for pyramid in self.pyramids:
            vus.append(pyramid.last_variation_unit())

        return vus

    def final_variation_unit(self) -> VariationalUnitData:
        return self.last_pyramid().last_variation_unit()

    def registered_images(self) -> List[tf.Tensor]:
        return [pl_data.registered_images() for pl_data in self.pyramids]

    def predicted_displacements(self) -> List[tf.Tensor]:
        return [pl_data.predicted_displacements() for pl_data in self.pyramids]

    def per_level_last_unit_registered_images(self) -> List[tf.Tensor]:
        return [vu_data.registered_images for vu_data in self.last_variation_units()]

    def per_level_last_unit_predicted_displacements(self) -> List[tf.Tensor]:
        return [
            vu_data.predicted_displacements for vu_data in self.last_variation_units()
        ]

    def clone(self) -> "PipeLineData":
        pl = PipeLineData()

        for pl_data in self.pyramids:
            pl.pyramids.append(pl_data.clone())

        return pl
