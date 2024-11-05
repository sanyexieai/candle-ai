# from ultralytics import YOLO

# import torch
# from safetensors.torch import load_file  # safetensors 库的加载函数

# # 加载 safetensors 文件
# safetensor_path = "yolov8s.safetensors"
# loaded_model = load_file(safetensor_path)

# # 转换为 PyTorch 模型格式
# torch.save(loaded_model, "yolov8s.pt")

# print("Conversion from safetensors to .pt completed.")



# import torch
# from safetensors.torch import save_file
# import json

# # 自定义序列化函数
# def custom_serializer(obj):
#     if isinstance(obj, torch.device):  # 例如处理 torch.device 对象
#         return str(obj)
#     # 可以添加更多类型的处理
#     return f"<non-serializable: {type(obj).__name__}>"

# 加载 PyTorch 模型
from ultralytics import YOLO

# Load your trained YOLO model
# model = YOLO("./yolov8s.pt")

# model = YOLO("yolov8s.pt") 
# model.export(format="onnx")
# # 过滤掉所有非 torch.Tensor 的值
# tensor_data = {k: v for k, v in model.items() if isinstance(v, torch.Tensor)}

# # 非 tensor 的部分
# non_tensor_data = {k: v for k, v in model.items() if not isinstance(v, torch.Tensor)}

# # 将模型的 tensor 部分保存为 safetensors 格式
# save_file(tensor_data, "yolov8s.safetensors")

# # 将非 tensor 的部分保存为 JSON 格式，使用自定义序列化函数
# with open("non_tensor_data.json", "w") as f:
#     json.dump(non_tensor_data, f, default=custom_serializer)

# print("Model tensors saved to yolov8s.safetensors and non-tensor data saved to non_tensor_data.json")




model = YOLO("yolov8s.onnx")  # 加载 YOLO 模型

# Run batched inference on a list of images
results = model(["bike.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk