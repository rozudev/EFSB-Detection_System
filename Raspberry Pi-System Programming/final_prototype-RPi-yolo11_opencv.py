from ultralytics import YOLO
from picamera2 import Picamera2
import cv2

model = YOLO("best2.1.pt")

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480),"format":"RGB888"}
)
picam2.configure(config)
picam2.start()

while True:

    frame = picam2.capture_array()

    results = model(
        frame,
        imgsz=640,
        conf=0.40,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("Proj F.R.O.N.T.I.E.R final prototype", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()