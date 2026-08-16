# PROJECT: EFSB Detection Research Prototype
# DEVELOPED BY: Fatima Rose P. Torres
# DESCRIPTION:
# This system utilizes a custom-trained YOLO11s (Small) model to detect Fruit and Shoot Borer infestation in eggplant farms.

import cv2 as cv
from ultralytics import YOLO
import pyfirmata2

comport = 'COM3'
board = pyfirmata2.Arduino(comport)

green_led = board.get_pin('d:6:o')
red_led = board.get_pin('d:5:o')

model = YOLO('best2.1.pt')

model.export(format="onnx", opset=12, simplify=True)

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int (frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

#read video
capture = cv.VideoCapture('eggplant farm video/lv_0_20260719150104.mp4')

while True:
    isTrue, frame = capture.read()

    if not isTrue:
        break

    results = model(frame, conf=0.20)
    annotated_frame = results[0].plot()

    count = 0
    for box in results[0].boxes:
        cls = int(box.cls[0])
        class_name = model.names[cls]

        if class_name in ['Fruit borer', 'Fruit rot']:
            count += 1
            red_led.write(1)
            green_led.write(0)


        else:
            red_led.write(0)
            green_led.write(1)


    frame_resized = rescaleFrame(annotated_frame, scale=0.6)

    cv.putText(
        frame_resized, f"EFSB Infestation: {count}", (15, 35),
        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3
    )

    cv.imshow('Research Prototype 2.1 (yolo11s)', frame_resized)

    if  cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()