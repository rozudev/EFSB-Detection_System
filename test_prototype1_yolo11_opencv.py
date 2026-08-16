from ultralytics import YOLO
model = YOLO('best1.pt')
model.export(format="onnx", opset=12, simplify=True)
results = model(source=0, show=True, conf=0.2, save=True)
