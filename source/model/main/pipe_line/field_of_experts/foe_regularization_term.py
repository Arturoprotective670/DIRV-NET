import tensorflow as tf

from source.model.main.pipe_line.field_of_experts.foe_operator import (
    FieldsOfExpertsOperator,
)
from source.model.main.learnable_variables import LearnableVariables
from source.model.settings.config import Config


class FoERegularizationTerm(FieldsOfExpertsOperator):
    """
    Purpose
    --------
    - Field of Experts class for the data term of DIRV-Net (see eq. 16).

    Contributors
    ------------
    - TMS-Namespace
    """

    def __init__(
        self,
        config: Config,
        learnable_variables: LearnableVariables,
        pyramid_level: int,
        variation_unit_index: int,
    ) -> None:
        super().__init__(config)

        self._py_index = pyramid_level
        self._vu_index = variation_unit_index

        self._vars = learnable_variables

    def Push(self, data: tf.Tensor) -> tf.Tensor:
        """
        Overrides base class method.
        """
        return super().Push(
            data,
            self._vars.regularization_term_kernels[self._py_index, self._vu_index, ...],
            self._vars.regularization_term_potential_function_parameters[
                self._py_index, self._vu_index, ...
            ],
        )
