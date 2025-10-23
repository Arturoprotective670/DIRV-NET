from typing import Any, Optional, Tuple, Union
from numpy.typing import NDArray
import numpy as np


class PreviewData:
    def __init__(
        self,
        grid_data: NDArray,
        titles: NDArray,
        map_ranges: Optional[Union[Tuple[float, float], NDArray[Any]]] = None,
        color_map_names: Optional[Union[str, NDArray[np.str_]]] = None,
        show_stats_boxes: Union[bool, NDArray[np.bool_]] = False,
        show_color_bar: bool = False,
        is_fields_stats: bool = False,
    ) -> None:
        self.grid_data: NDArray = grid_data

        self.titles: NDArray = titles

        self.map_ranges: Union[Tuple[float, float], NDArray[Any]] = map_ranges

        self.color_map_names: Union[str, NDArray[np.str_]] =  color_map_names

        self.show_stats_boxes: Union[bool, NDArray[np.bool_]] = show_stats_boxes

        self.show_color_bar: bool = show_color_bar

        self.is_fields_stats: bool = is_fields_stats
