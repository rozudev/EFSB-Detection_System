#============================================================================================================
# PROJECT: EFSB Detection Research Prototype
# DEVELOPED BY: Fatima Rose P. Torres
# DESCRIPTION:
# This system utilizes a custom-trained YOLO26 Nano model to detect Fruit and Shoot Borer in eggplant farms.
#============================================================================================================

import cv2 as cv
from ultralytics import YOLO

model = YOLO('best2.pt')

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int (frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

#read video
capture = cv.VideoCapture('eggplant farm video/lv_0_20260410231625.mp4')

while True:
    isTrue, frame = capture.read()

    if not isTrue:
        break

    results = model(frame, conf=0.25)
    annotated_frame = results[0].plot()

    frame_resized = rescaleFrame(annotated_frame, scale=0.6)

    cv.imshow('Research Prototype 2', frame_resized)

    if  cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
