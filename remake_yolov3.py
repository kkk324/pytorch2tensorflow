import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import cv2

## 0 convert
#convert(cfg='cfg/yolov3.cfg', weights='weights/yolov3.weights')
# output path = ''weights/yolov3.pt'


################
## 1. load model
###############
img_size = (416, 416)
new_cfg     = 'cfg/yolov3_no_yolo_layer.cfg'
cfg         = 'cfg/yolov3.cfg'
model       = Darknet(cfg, img_size)
#new_model   = Darknet(new_cfg, img_size)

#for param_tensor in new_model.state_dict():
#    print(param_tensor, "\t", new_model.state_dict()[param_tensor].size())
#print(type(model)) #<class 'models.Darknet'>
#exit()

############################
## 2. load weight from model
############################
weights = 'weights/yolov3.pt'
#print((torch.load(weights)['model'].keys()))
#for k in torch.load(weights)['model'].keys():
#          print(k)

#checkpoint = torch.load(weights)
#print(checkpoint.keys())
#new_model.load_state_dict(checkpoint['model'])
#new_model.eval()

#print('type of checkpoint:', type(checkpoint))
#for k, v in checkpoint['model'].items():
#    num_node      =  int(k.split('.')[1])
#    name_node     =  k.split('.')[2] 
#    name_paramter =  k.split('.')[3]   
#    print('num_node, name_node, name_paramter', num_node, name_node, name_paramter)
    #if num_node <= 81:
    #    pass
        #print('k:', num_node, name_node)
    #if num_node is 84 and name_node == 'Conv2d':
    #    print('k:', num_node, name_node, )

#exit()

device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
model.load_state_dict(torch.load(weights, map_location=device)['model'])
model.to(device).eval()
#new_model.load_state_dict(torch.load(weights, map_location=device)['model'])
#new_model.to(device).eval()
#print('Done.')

################
## 3 save onnx
#################
import torch
model_name = 'weights/yolov3.onnx'
#print(model_name)
#print('img_size: ', img_size)
img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
torch.onnx.export(model, img, model_name, verbose=True, opset_version=10)

##########################
## Validate exported model
##########################
#import onnx
#onnx_model = onnx.load(model_name)  # Load the ONNX model
#onnx.checker.check_model(onnx_model)  # Check that the IR is well formed
#print('-----graph-----')
#print(onnx.helper.printable_graph(onnx_model.graph))  # Print a human readable representation of the graph





## 3. show every nodes as order
#import onnx
#model = onnx.load('weights/yolov3.onnx')
#onnx.checker.check_model(model)
#onnx.helper.printable_graph(model.graph)

#print('---3. run onnx model on onnx runtime---')
#import onnxruntime as ort
#import numpy as np

 
#ort_session = ort.InferenceSession('weights/yolov3.onnx')
#outputs = ort_session.run(None, {"actual_input_1":np.zeros((1,3)+img_size).astype(np.float32)})


