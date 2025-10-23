import copy
from typing import Any, List
from source.framework.tools.general import StatisticsData
from source.model.main.pipe_line.pipe_line_data import PipeLineData


class RecurrentRefinementData:
  """
  Purpose
  -------
  A class to hold information about the loss and statistics of the
  recurrent refinement process.

  Notes
  -----
  - The loss is approximate, since we have GT fields only from the moving
    to fixed images, and for intermediate steps, we calc recurrent GT
    fields, by subtracting the new predicted fields from the original Gt
    fields, which is not fully precise.
  - For same above reason, iteration index could be an average of the
    iterations count.

  Contributors
  ------------
  - TMS-Namespace
  """
  def __init__(
      self,
      pipeline_data: PipeLineData,
      pseudo_loss: float,
      statistics: StatisticsData,
      extras: List[Any],
      pseudo_iteration: float,
  ) -> None:
      self.pipeline_data = pipeline_data
      self.statistics = statistics

      self.pseudo_loss = pseudo_loss
      self.pseudo_iteration = pseudo_iteration

      self.extras_from_loss_function = extras

  def clone(self) -> "RecurrentRefinementData":
      cp = copy.copy(self)
      cp.pipeline_data = self.pipeline_data.clone()
      cp.statistics = self.statistics.clone()
      cp.extras_from_loss_function = [self.extras_from_loss_function[0].clone()]

      return cp
