"""
Purpose
-------
- Check if image dimensions satisfies the DIRV-Net (mostly
technical) needs (related to pyramid generation, and using only integer scale
factors in resizing).

Contributors
------------
- TMS-Namespace
"""

from typing import List, Tuple, Union


def is_legit_shape(
    shape: Union[Tuple[int, int], Tuple[int, int, int]],
    pyramid_levels_count: int,
    displacements_control_points_spacings: Union[Tuple[int, int], Tuple[int, int, int]],
    raise_exception: bool,
) -> bool:
    """
    Checks if all dimensions of the given shape satisfies the VatNet' needs.
    """
    for dim in range(len(shape)):
        if (
            is_legit_dimension_size(
                shape[dim],
                pyramid_levels_count,
                displacements_control_points_spacings[dim],
                raise_exception,
            )
            is False
        ):
            return False

    return True


def is_legit_dimension_size(
    size: int,
    pyramid_levels_count: int,
    displacements_control_points_spacing: int,
    raise_exception: bool = False,
) -> bool:
    """
    Checks if the given size satisfies the VatNet' needed size.
    """
    # Since resizing can be done only of an integer scale factor, because we are
    # stride using convolution for that, we need to make sure that image size is
    # sufficient to generate the required amount of pyramids of integer size
    if size % 2 ** (pyramid_levels_count - 1) != 0:
        if raise_exception:
            raise Exception(
                "Can't generate the requested amount of "
                + "pyramids of integer image size."
            )
        else:
            return False

    smallest_pyramid_size = size / 2 ** (pyramid_levels_count - 1)

    # check if the smallest pyramid resolution too small too contain enough
    # control points count
    MIN_CONTROL_POINTS_COUNT = (
        2  # at least two control points should fit in a dimension
    )

    if (
        smallest_pyramid_size
        < displacements_control_points_spacing * MIN_CONTROL_POINTS_COUNT
    ):
        if raise_exception:
            raise Exception(
                "Control points spacings too big for " + "smallest pyramid level size."
            )
        else:
            return False

    # An integer number of control points should fit in smallest pyramid
    if smallest_pyramid_size % displacements_control_points_spacing != 0:
        if raise_exception:
            raise Exception(
                "Control points spacings are not compatible with"
                + "smallest level pyramid size."
            )
        else:
            return False

    if pyramid_levels_count > 1:
        second_smallest_pyramid_size = smallest_pyramid_size * 2
        smallest_pyramid_displacements_grid_size = (
            smallest_pyramid_size / displacements_control_points_spacing
        )

        # because we will need to upsample the displacement field, and this can
        # be done only by integer scale factor (due to the use of strided
        # convolution), so upsampling to the size of the smallest pyramid, we
        # need to make sure, that initial displacements grid size is a multiple
        # of the second smallest pyramid size
        if second_smallest_pyramid_size % smallest_pyramid_displacements_grid_size != 0:
            if raise_exception:
                raise Exception(
                    "Control points spacings are not compatible"
                    + " with second smallest level pyramid size."
                )
            else:
                return False

    return True
