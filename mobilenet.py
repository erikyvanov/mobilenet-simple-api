from typing import List
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils


class Mobilenet:
    def __init__(self) -> None:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2()

    def _prepare_ndarray_image(self, img: np.ndarray):
        img_array = image.img_to_array(img)
        img_array = image.smart_resize(img_array, size=(224, 224))
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)

    def _parse_results(self, results: List) -> List:
        parse_results = []
        results = results[0]

        for result in results:
            parse_result = {
                'name': result[1],
                'probability': float(result[2])
            }

            parse_results.append(parse_result)

        return parse_results

    def predict(self, img: np.ndarray):
        preprocessed_image = self._prepare_ndarray_image(img)
        predictions = self.mobilenet.predict(preprocessed_image)

        results = imagenet_utils.decode_predictions(predictions)

        return self._parse_results(results)
