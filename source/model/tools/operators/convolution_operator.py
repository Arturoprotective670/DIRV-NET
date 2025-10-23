import tensorflow as tf


class ConvolutionOperator:
    def __init__(self, rank: int):
        if rank == 2:
            self.convolve = tf.nn.conv2d
            self.convolve_transpose = tf.nn.conv2d_transpose

        elif rank == 3:
            self.convolve = tf.nn.conv3d
            self.convolve_transpose = tf.nn.conv3d_transpose

        else:
            raise NotImplementedError
