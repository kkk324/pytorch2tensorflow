import tensorflow as tf
import torch
import torchvision

import numpy
import onnx
import onnxruntime as rt
from onnx_tf.backend import prepare


print('torch:           ', torch.__version__)
print('torchvision:     ', torchvision.__version__)
print('onnx:            ', onnx.__version__)
print('onnx-tf:         ', '1.5.0 (pip install onnx-tf==1.5.0)')
print('onnxruntime-gpu: ', rt.__version__)
print('onnxruntime:     ', rt.__version__)
print('tensorflow:      ', tf.__version__)
