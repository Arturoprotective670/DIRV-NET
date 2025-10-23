from enum import IntEnum


class PotentialsParametrizationModes(IntEnum):
    PIECE_WISE_LINEAR = 0
    RBF = 1


class OptimizationApproaches(IntEnum):
    GRADIENT_DESCENT = 0
    POLYAK_HEAVY_BALL = 1
