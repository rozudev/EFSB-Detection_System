from ultralytics import YOLO
model = YOLO('best2.pt')
model.export(format="onnx", opset=12, simplify=True)
results = model(source=0, show=True, conf=0.25, save=True)