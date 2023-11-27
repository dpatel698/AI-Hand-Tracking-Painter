# Citation: https://github.com/kinivi/hand-gesture-recognition-mediapipe
import numpy as np
import tensorflow as tf


class HandSignClassifier(object):
    def __init__(
        self,
        model_path='model/handSignClassifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.inputs = self.interpreter.get_input_details()
        self.outputs = self.interpreter.get_output_details()

    def __call__(
        self,
        landmarks,
    ):
        input_details_tensor_index = self.inputs[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmarks], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.outputs[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_idx = np.argmax(np.squeeze(result))

        return result_idx