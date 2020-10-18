import tensorflow as tf

in_path = "../models/mnv2.pb"
#in_path = '20200427_112613.pb'
#out_path = "hrglass_256_64.tflite"
out_path = "mnv2.tflite"

# 模型输入节点
input_tensor_name = ["Image"]
input_tensor_shape = {"Image":[1,224,224,3]}
# 模型输出节点
classes_tensor_name = ["output"]

#For Version below 2.0
converter = tf.lite.TFLiteConverter.from_frozen_graph(in_path, input_tensor_name, classes_tensor_name, input_shapes=input_tensor_shape)

#For Version above 2.0
#converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(in_path, input_tensor_name, classes_tensor_name, input_shapes=input_tensor_shape)

#converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]

#converter.post_training_quantize = True
tflite_model = converter.convert()

with open(out_path, "wb") as f:
    f.write(tflite_model)