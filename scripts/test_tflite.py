import numpy as np
import tensorflow as tf
import os

from pathlib import Path
from PIL import Image

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

label_file = "models/retrained/retrained_labels.txt"

labels = load_labels(label_file)
xy = []
correct = 0

image_dir = "data/test"
for child in [name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]:
    key = os.path.basename(child)
    for p in list((Path(image_dir) / child).glob("*.jpg"))[:3]:
      t = np.array(Image.open(p).resize(input_shape[-3:-1]), dtype=np.float32).reshape(input_shape) / 255
      key = key.replace("_", " ")
      xy.append((t, key))


correct = 0
# Test model on random input data.
for x, y in xy:
    interpreter.set_tensor(input_details[0]['index'], x)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Expecting: {} \t Got: {}".format(y,labels[output_data.argmax()]))
    if labels[output_data.argmax()] == y:
        correct += 1

print("Test Accuracy: {}%".format((correct / len(xy)*100)))