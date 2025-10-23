from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import tensorflow as tf

from source.model.main.pipe_line.pipe_line_data import PipeLineData


class SurrogateLossInterface(ABC):
    @abstractmethod
    def loss(
        self,
        pip_line_data: PipeLineData,
        args: Optional[List[Any]] = None,
    ) -> Tuple[tf.Tensor, List[Any]]:
        """
        Calculates the losses of the forward propagated images.

        Arguments
        ---------
        - `pip_line_data`: PipeLineData object that holds information of all
          pipeline stages.
        - `args`: a list of variables that we what to pass to loss function.

        Notes
        -----
        - The returned loss value should be necessary a tensor, because it will
          br tracked by TF.

        Returns
        -------
        - The loss value.
        - A list of any other variables that we want to pass forward.
        """
        pass
