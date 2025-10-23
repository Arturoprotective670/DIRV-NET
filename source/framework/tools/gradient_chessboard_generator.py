"""
Purpose
-------
- Generates ChessBoard like 2D/3D images.
- Board colors can be changed in a gradient manner.
- Generate boards of random attributes.

Contributors
------------
- TMS-Namespace
"""

from typing import List, Optional, Tuple, Union
from numpy.typing import NDArray

import numpy as np


def random_gradient_chessboard(
    output_size: Union[Tuple[int, int], Tuple[int, int, int]],
    random_cells_size_range: Tuple[int, int],
    channels_count: int,
    random_generator: np.random.Generator = np.random.default_rng(),
) -> NDArray[np.uint8]:
    rank = len(output_size)
    corners_count = 2**rank

    def rnd() -> int:
        return random_generator.integers(0, 255)

    even = [[rnd() for _ in range(channels_count)] for _ in range(corners_count)]
    odd = [[rnd() for _ in range(channels_count)] for _ in range(corners_count)]

    cell_size = random_generator.choice(
        range(random_cells_size_range[0], random_cells_size_range[1])
    )

    return gradient_chessboard(output_size, [cell_size] * rank, even, odd)


def gradient_chessboard(
    output_size: Union[Tuple[int, int], Tuple[int, int, int]],
    cells_size: Union[Tuple[int, int], Tuple[int, int, int]],
    corners_even_colors: List[Tuple[int]],
    corners_odd_colors: Optional[List[Tuple[int]]] = None,
) -> NDArray[np.uint8]:
    """
    Generates an `array`, that represents an image of chess-board-like structure,
    and such that the rectangles are colored with gradient colors.

    Arguments
    ---------
    - `output_size` : The height and width of the needed image.
    - `cells_size`: The needed size of the rectangles
    in height and width dimensions.
    - `corners_even_colors`: a `list` of 4/8 (for 2d and 3d cases respectfully)
    3-`tuples` in the range [0, 255], that represents the RGB colors at each
    corner of the image, for 2d case it will be `(top left, top right,
    bottom left, bottom right)`, for the even index of the rectangles.
    - `corners_odd_colors`: same as previous argument, but for odd indexes of
    the rectangles.

    Notes
    -----
    - If `corners_odd_colors` is `None`, it will be considered to be equal to
    `corners_even_colors`.
    - The number of channels in the output image array, is defined by the number
    of color channels that been provided by the first corner of an even index.
    - If the requested size of cells dose not divide the requested image
    size, pixels will be filled with the previous last cell colors.
    """

    rank = len(output_size)

    if rank == 2:
        return _gradient_chessboard_2d(
            output_size, cells_size, corners_even_colors, corners_odd_colors
        )
    elif rank == 3:
        return _gradient_chessboard_3d(
            output_size, cells_size, corners_even_colors, corners_odd_colors
        )
    else:
        raise Exception("Unsupported requested image rank.")


def _gradient_chessboard_2d(
    output_size: Tuple[int, int],
    cells_size: Tuple[int, int],
    corners_even_colors: List[Tuple[int]],
    corners_odd_colors: List[Tuple[int]] = None,
) -> NDArray[np.uint8]:
    channels_count = len(corners_even_colors[0])
    img = np.zeros([*output_size, channels_count], dtype=np.uint8)
    corners_even_colors = np.array(corners_even_colors, dtype=np.uint8)

    if corners_odd_colors is None:
        corners_odd_colors = corners_even_colors
    else:
        corners_odd_colors = np.array(corners_odd_colors, dtype=np.uint8)

    y_cells_size = cells_size[0]
    x_cells_size = cells_size[1]

    y_cells_count = output_size[0] // y_cells_size
    x_cells_count = output_size[0] // x_cells_size

    for i in range(x_cells_count):
        for j in range(y_cells_count):
            x1 = i * x_cells_size
            y1 = j * y_cells_size

            if i == x_cells_count - 1:
                x2 = output_size[1]
            else:
                x2 = (i + 1) * x_cells_size

            if j == y_cells_count - 1:
                y2 = output_size[0]
            else:
                y2 = (j + 1) * y_cells_size

            w1 = (x_cells_count - i - 1) * (y_cells_count - j - 1)
            w2 = i * (y_cells_count - j - 1)
            w3 = (x_cells_count - i - 1) * j
            w4 = i * j

            weights = np.array([w1, w2, w3, w4])
            weights = np.stack([weights / np.sum(weights)] * channels_count, axis=1)

            if (i + j) % 2 == 0:
                corner_colors = corners_even_colors
            else:
                corner_colors = corners_odd_colors

            img[y1:y2, x1:x2, :] = np.sum(
                corner_colors * weights, axis=0, dtype=np.uint8
            )

    return img


def _gradient_chessboard_3d(
    output_size: Tuple[int, int, int],
    cells_size: Tuple[int, int, int],
    corners_even_colors: List[Tuple[int]],
    corners_odd_colors: List[Tuple[int]] = None,
) -> NDArray[np.uint8]:
    channels_count = len(corners_even_colors[0])
    img = np.zeros([*output_size, channels_count], dtype=np.uint8)
    corners_even_colors = np.array(corners_even_colors, dtype=np.uint8)

    if corners_odd_colors is None:
        corners_odd_colors = corners_even_colors
    else:
        corners_odd_colors = np.array(corners_odd_colors, dtype=np.uint8)

    z_cells_size = cells_size[2]
    y_cells_size = cells_size[0]
    x_cells_size = cells_size[1]

    z_cells_count = output_size[2] // z_cells_size
    y_cells_count = output_size[0] // y_cells_size
    x_cells_count = output_size[1] // x_cells_size

    for k in range(z_cells_count):
        for i in range(x_cells_count):
            for j in range(y_cells_count):
                x1, y1, z1 = i * x_cells_size, j * y_cells_size, k * z_cells_size

                if i == x_cells_count - 1:
                    x2 = output_size[1]
                else:
                    x2 = (i + 1) * x_cells_size

                if j == y_cells_count - 1:
                    y2 = output_size[0]
                else:
                    y2 = (j + 1) * y_cells_size

                if k == z_cells_count - 1:
                    z2 = output_size[2]
                else:
                    z2 = (k + 1) * z_cells_size

                w1 = (
                    (x_cells_count - i - 1)
                    * (y_cells_count - j - 1)
                    * (z_cells_count - k - 1)
                )
                w2 = i * (y_cells_count - j - 1) * (z_cells_count - k - 1)
                w3 = (x_cells_count - i - 1) * j * (z_cells_count - k - 1)
                w4 = i * j * (z_cells_count - k - 1)
                w5 = (x_cells_count - i - 1) * (y_cells_count - j - 1) * k
                w6 = i * (y_cells_count - j - 1) * k
                w7 = (x_cells_count - i - 1) * j * k
                w8 = i * j * k

                weights = np.array([w1, w2, w3, w4, w5, w6, w7, w8])
                weights = np.stack([weights / np.sum(weights)] * channels_count, axis=1)

                if (i + j + k) % 2 == 0:
                    corner_colors = corners_even_colors
                else:
                    corner_colors = corners_odd_colors

                img[y1:y2, x1:x2, z1:z2, :] = np.sum(
                    corner_colors * weights, axis=0, dtype=np.uint8
                )

    return img
