import torch
import torchvision

#dummy_input = torch.randn(10, 3, 224, 224, device='cpu')

img_size = (224, 224)
dummy_input = torch.zeros((10, 3) + img_size)
model = torchvision.models.vgg16(pretrained=True).cpu()

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.299,0.224,0.225]
        )])
    

from PIL import Image
img = Image.open("dog.jpg")

img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

model.eval()
out = model(batch_t)
print(out.shape)

with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0]*100
print(classes[index[0]], percentage[index[0]].item())
print('--end---')
exit()

input_name = ["actual_input_1"] + ["learned_%d" % i for i in range(16) ]
output_names = ["output1"]

print(input_names)
print(output_names)

torch.onnx.export(model, dummy_input, "./weights/alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names,opset_version=10)

print('---1---')

import onnx
model = onnx.load("./weights/alexnet.onnx")
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)


print('---2---')
import onnxruntime as ort
import numpy as np
ort_session = ort.InferenceSession('./weights/alexnet.onnx')

outputs = ort_session.run(None, {"actual_input_1":np.zeros((10,3)+(224,224)).astype(np.float32)})

print(outputs[0])


