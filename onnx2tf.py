import onnx 
from onnx_tf.backend import prepare  
path = './weights/'
model_name = 'alexnet'


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
