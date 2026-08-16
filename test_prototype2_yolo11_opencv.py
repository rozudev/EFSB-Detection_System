from ultralytics import YOLO
model = YOLO('best2.1.pt') # you can change it to best2.1.pt or best2.1.pt
model.export(format="onnx", opset=12, simplify=True)
results = model(source=0, show=True, conf=0.25, save=True)