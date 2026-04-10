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

for t in tensor_details:
    print("=" * 50)
    print("name:", t['name'])
    print("index:", t['index'])
    print("dtype:", t['dtype'])

    qp = t.get('quantization_parameters', None)
    if qp:
        print("scales:", qp['scales'])
        print("zero_points:", qp['zero_points'])
        print("quant_dim:", qp['quantized_dimension'])
print(interpreter.get_tensor(output_details[0]['index']))
# for d in tensor_details:
#     print(d['index'], d['name'], d['shape'])
# output = interpreter.get_tensor(4)
# print(output)