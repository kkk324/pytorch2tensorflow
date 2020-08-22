import torch
import torchvision

from utils.datasets import *
from utils.utils import *
### input size and zero input 
### model name and where to save model

#dummy_input = torch.randn(10, 3, 224, 224, device='cpu')
############################
model_name = 'shufflenet_v2_x1_0'
###########################

if model_name == '':
    img_size = (416, 416)
else:
    img_size = (224, 224)
dummy_input = torch.zeros((1, 3) + img_size)

save_onnx = './weights/' + model_name + '.onnx'
print(save_onnx)

if model_name == 'yolov3':
    img_size = (416, 416)
else:
    img_size = (224, 224)
if model_name == 'resnet18':
        img_size = (224, 224)
        model = torchvision.models.resnet18(pretrained=True).cpu()
elif model_name == 'squeezenet':
        img_size = (224, 224)
        model = torchvision.models.squeezenet1_0(pretrained=True).cpu()
elif model_name == 'alexnet':
        img_size = (224, 224)
        model = torchvision.models.alexnet(pretrained=True).cpu()
elif model_name == 'mobilenet_v2':
        img_size = (224, 224)
        model = torchvision.models.mobilenet_v2(pretrained=True).cpu()
elif model_name == 'inception_v3':
        img_size = (224, 224)
        model = torchvision.models.inception_v3(pretrained=True).cpu()
elif model_name == 'densenet161':
        img_size = (224, 224)
        model = torchvision.models.densenet161(pretrained=True).cpu()
elif model_name == 'shufflenet_v2_x1_0':
        img_size = (224, 224)
        model = torchvision.models.densenet161(pretrained=True).cpu()
elif model_name == 'yolov3':
        weights = 'weights/yolov3.weights'
        cfg     = 'cfg/yolov3.cfg'
        from models import *
        img_size = (416, 416)
        model = Darknet(cfg, img_size)
        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)


input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16) ]
output_names = ["output1"]

torch.onnx.export(model, dummy_input, save_onnx, verbose=True, input_names=input_names, output_names=output_names,opset_version=10)

print('---1. load onnx model and check it.---')
import onnx
model = onnx.load(save_onnx)
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)


print('---2. run onnx model on onnx runtime---')
import onnxruntime as ort
import numpy as np

if model_name == 'yolov3': 
    ort_session = ort.InferenceSession(save_onnx)
    img_size = (416, 416)
    outputs = ort_session.run(None, {"actual_input_1":np.zeros((1,3)+img_size).astype(np.float32)})
else:
    ort_session = ort.InferenceSession(save_onnx)
    img_size = (224, 224)
    outputs = ort_session.run(None, {"actual_input_1":np.zeros((1,3)+img_size).astype(np.float32)})

### save output 
import os
path = os.getcwd()
print(path)
print(outputs[0][0])
np.savetxt(model_name + '_onnx_output.txt', outputs[0][0])

#print('onnx output:')
#print(outputs[0].shape)
#print(outputs[0])

### load and check pb
import onnx 
from onnx_tf.backend import prepare  
path = './weights/'

onnx_path = path + model_name + '.onnx'
print(onnx_path)

onnx_model = onnx.load(onnx_path)  # load onnx model 
# output = prepare(onnx_model).run(input)  # run the loaded model
# no strict to be faster
output = prepare(onnx_model, strict=True)

pb_path = path + model_name + '.pb'
print(pb_path)
file = open(pb_path, "wb")
file.write(output.graph.as_graph_def().SerializeToString())
file.close()

print('---Your model pb is ready ---') 

import tensorflow as tf
import cv2
import numpy as np

graph_def = pb_path

with tf.gfile.GFile(graph_def, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())

n = len(restored_graph_def.node)
#print(n)
k = 0
for node in restored_graph_def.node:
    #if 'output' in node.name.lower():
    print(node.name, node.op)
    if k == n-1:
        output_node = node.name
        #print(node.name, node.op)
    k+=1

tf.import_graph_def(
    restored_graph_def,
    input_map=None,
    return_elements=None,
    name="")

#img = cv2.imread('data/samples/bus.jpg')
#img = cv2.resize(img, (416, 416))
#img = img[None, :, :, :]
#img = np.transpose(img, [0, 3, 1, 2])

if model_name == 'yolov3':
    img_size = (416, 416)
else:
    img_size = (224, 224)
img = np.zeros((1,3)+img_size).astype(np.float32)
#img = np.transpose(img, [0,3,1,2])

with tf.Session() as sess:
    pred = sess.run(node.name + ":0", feed_dict={'actual_input_1:0':img})

print(pred)
np.savetxt(model_name + '_pb_output.txt', pred)

