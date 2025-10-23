"""
Purpose
-------
- Various enums for the framework.

Contributors
------------
- TMS-Namespace
"""

from enum import IntEnum


class Stages(IntEnum):
    """
    An enum that determines the `DataSet` type, or more particularly, at which
    stage of workflow this data should be used.
    """

    TRAINING = 0
    VALIDATION = 1
    TESTING = 2
    INFERRING = 3


class SyntheticFields(IntEnum):
    TRUNCATED_WHITE_NOISE = 0
    AFFINE_FIELDS = 1
    NON_RIGID_FIELDS = 2
    COMBINED = 3


class SurrogateLossFunction(IntEnum):
    LAST_PYRAMID = 0
    PER_PYRAMID_LAST_VARIATIONAL_UNIT = 1


class PreProcessingMethod(IntEnum):
    NONE = 0
    GLOBAL_UNITY_NORMALIZATION = 1
    LOCAL_UNITY_NORMALIZATION = 2  # per image normalization
    RESIZE = 3


class DataSets(IntEnum):
    FASHION_MINST = 0
    NRRD_SLICED = 1
    PATCHED_256_MONOCHROME_ANHIR = 2
    SLICED_4DCT = 3
    INTERPOLATED_4DCT = 4
    GRADIENT_CHESSBOARD_64 = 5
    GRADIENT_CHESSBOARD_128 = 6
    GRADIENT_CHESSBOARD_64_DENSE = 7
