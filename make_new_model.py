import torch
import torchvision

from utils.datasets import *
from utils.utils import *
import densenet_partial as dp

img_size = (14, 14)
#model = torchvision.models.densenet121(pretrained=True).cpu()
pmodel = dp.densenet121_part(pretrained=True).cpu()

#print(model)
#print((pmodel))


for w in pmodel.state_dict():
    print(w)

model_name = 'weights/densenet121_part.onnx'
img = torch.zeros((1, 1024) + img_size)  # (1, 3, 320, 192)
torch.onnx.export(pmodel, img, model_name, verbose=True, opset_version=10)



#for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

#print(model.load_state_dict(model.state_dict()))

import onnx 
from onnx_tf.backend import prepare  
path = './weights/'

onnx_path = 'weights/densenet121_part.onnx'
print(onnx_path)

onnx_model = onnx.load(onnx_path)  # load onnx model 
# output = prepare(onnx_model).run(input)  # run the loaded model
# no strict to be faster
output = prepare(onnx_model, strict=False)

pb_path = path + 'densenet121_part.pb'
print(pb_path)
file = open(pb_path, "wb")
file.write(output.graph.as_graph_def().SerializeToString())
file.close()

print('---Your model pb is ready ---') 
