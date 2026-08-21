from ultralytics import YOLO

# Load pretrained yolo11 model
model = YOLO('best1.pt')

#convert best.pt to torch.onnx format
model.export(format="onnx")
