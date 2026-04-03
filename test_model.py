import tensorflow as tf
import model
from PIL import Image
import numpy as np


input = Image.open("samples/yes_sample_features_int8.pgm")
input = np.array(input).reshape(1, 49, 10, 1)

interpreter = tf.lite.Interpreter(model_path="model/KWS_ds_cnn_s_quant_power.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
tensor_details = interpreter.get_tensor_details()
interpreter.set_tensor(input_details[0]['index'], input)
interpreter.invoke()
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)
details = interpreter.get_tensor_details()

for t in details:
    print(t["name"])
    print("scale:", t["quantization"][0])
    print("zero_point:", t["quantization"][1])
    print("dtype:", t["dtype"])
    print()