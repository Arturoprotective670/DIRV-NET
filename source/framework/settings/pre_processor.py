from typing import Callable
import tensorflow as tf

from source.framework.settings.whole_config import WholeConfig
from source.framework.settings.enums import PreProcessingMethod


class PreProcessor:
    """
    Purpose
    -------
    - Pre-Process image batches depending of the config file settings.

    Contributors
    ------------
    - TMS-Namespace
    """
    def __init__(self, config: WholeConfig) -> None:
        self._config = config

        self.pre_process_images: Callable[[tf.Tensor], tf.Tensor]

        if config.image_pre_processing_method == PreProcessingMethod.NONE:
            self.pre_process_images = self._process_none
        elif (
            config.image_pre_processing_method
            == PreProcessingMethod.LOCAL_UNITY_NORMALIZATION
        ):
            self.pre_process_images = self._process_local_unity
        elif (
            config.image_pre_processing_method
            == PreProcessingMethod.GLOBAL_UNITY_NORMALIZATION
        ):
            self.pre_process_images = self._process_global_unity
        elif config.image_pre_processing_method == PreProcessingMethod.RESIZE:
            self.pre_process_images = self._process_rescale
        else:
            raise NotImplementedError

    def _process_none(self, batch_images: tf.Tensor) -> tf.Tensor:
        return batch_images

    def _process_local_unity(self, batch_images: tf.Tensor) -> tf.Tensor:
        assert tf.reduce_max(batch_images).numpy() > 1

        if batch_images.dtype in [tf.int8, tf.int16, tf.int32, tf.int64]:
            batch_images = tf.cast(
                batch_images, self._config.float_data_type
            )

        images_tensors = tf.unstack(batch_images, axis=0)

        for index, image in enumerate(images_tensors):
            images_tensors[index] = image / tf.reduce_max(image)

        return tf.stack(images_tensors, axis=0)

    def _process_global_unity(self, batch_images: tf.Tensor) -> tf.Tensor:
        assert tf.reduce_max(batch_images).numpy() > 1

        if batch_images.dtype in [tf.int8, tf.int16, tf.int32, tf.int64]:
            batch_images = tf.cast(
                batch_images, self._config.float_data_type
            )

        return batch_images / 255

    def _process_rescale(self, batch_images: tf.Tensor) -> tf.Tensor:
        # TODO: upgrade to use resizing operator
        return tf.image.resize(
            batch_images, self._config.resized_image_size, method="bilinear"
        )
