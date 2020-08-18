import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import cv2

## 0 convert
#convert(cfg='cfg/yolov3.cfg', weights='weights/yolov3.weights')

## 1. load model 
img_size = (416, 416)
cfg     = 'cfg/yolov3_no_yolo_layer.cfg'
#cfg     = 'cfg/yolov3.cfg'
model = Darknet(cfg, img_size)

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

exit()

## 2. load weight
weights = 'weights/yolov3.pt'
#w = load_darknet_weights(model, weights)

print(type(torch.load(weights)))
print(len(torch.load(weights)))
#print((torch.load(weights)['model'].keys()))
for k in torch.load(weights)['model'].keys():
          print(k)
exit()

model.load_state_dict(torch.load(weights))
model.eval()
exit()

#device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
#model.load_state_dict(torch.load(weights, map_location=device)['model'])
#model.to(device).eval()


















## 3. show every nodes as order
import onnx
model = onnx.load('weights/yolov3.onnx')
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)

print('---3. run onnx model on onnx runtime---')
import onnxruntime as ort
import numpy as np

 
ort_session = ort.InferenceSession('weights/yolov3.onnx')
outputs = ort_session.run(None, {"actual_input_1":np.zeros((1,3)+img_size).astype(np.float32)})


