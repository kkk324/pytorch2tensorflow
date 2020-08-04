import tensorflow as tf
import cv2
import numpy as np


graph_def = 'weights/resnet18.pb'

with tf.gfile.GFile(graph_def, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())

n = len(restored_graph_def.node)
print(n)
k = 0
for node in restored_graph_def.node:
    #if 'output' in node.name.lower():
    if k == n-1:
        output_node = node.name
        print(node.name, node.op)
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

#exit()

img = np.zeros((10,3)+(224,224)).astype(np.float32)
#img = np.transpose(img, [0,3,1,2])

with tf.Session() as sess:
    pred = sess.run(output_node + ":0", feed_dict={'actual_input_1:0':img})

print(pred)

# writer = tf.summary.FileWriter("./graph", graph)     
# writer.close()


