import torch
import torchvision

from utils.datasets import *
from utils.utils import *
import densenet_partial as dp

img_size = (224, 224)
#model = torchvision.models.densenet121(pretrained=True).cpu()
pmodel = dp.densenet121_part(pretrained=True).cpu()

#print(model)
#print((pmodel))


for w in pmodel.state_dict():
    print(w)

model_name = 'weights/densenet121_part.onnx'
img = torch.zeros((1, 32) + img_size)  # (1, 3, 320, 192)
torch.onnx.export(pmodel, img, model_name, verbose=True, opset_version=10)

exit()

#for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

#print(model.load_state_dict(model.state_dict()))

