"""Face detection script.

This script allows to check the face detection model used in reachy_face_tracking
by opening an opencv window with the result of the detection model inference.
This can help debugging the application by checking if the detection is actually
working. You can also use it to adjust the tracking_threshold variable used in the
application to decide whether a face detected is close enough from the robot.

To call this script:
    cd ~/dev/reachy-face-tracking
    python3 -m reachy_face_tracking.test_face_detection
"""
import logging
from pathlib import Path

import cv2 as cv
from reachy_sdk import ReachySDK
from .detection import Detection

logger = logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

model_path = str(Path.cwd() / 'models' / 'ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite')


class FaceDetectionTester:
    def __init__(self) -> None:
        self.reachy = ReachySDK(host='localhost')
        self.detection = Detection(self.reachy, detection_model_path=model_path)
        self.tracking_threshold = 20 * 20  # Threshold actualy used in the application

    def get_annotated_image(self) -> None:
        im = self.detection._image

        if not self.detection._somebody_detected:
            return im

        xmin, ymin, xmax, ymax, size = self.detection._face_emb

        # If someone is detected, but too far, cancel the detection
        if size < self.tracking_threshold:
            return im

        # The coordinates are in the frame of the resized image (320, 320)
        # To display the bounding box in the original frame (640, 480)
        # we need to multiply by the scale factor
        scale_x = 480 / 320
        scale_y = 640 / 320

        cv.rectangle(
            im,
            (int(xmin * scale_x), int(ymin * scale_y)),
            (int(xmax * scale_x), int(ymax * scale_y)),
            (255, 0, 0),
            2,
            )
        return im


if __name__ == '__main__':
    face_detection_tester = FaceDetectionTester()

    face_detection_tester.detection.start()

    import time
    time.sleep(2.0)

    while True:
        im = face_detection_tester.get_annotated_image()
        cv.imshow('Face detection.', im)
        if cv.waitKey(33) & 0xFF == ord('q'):
            face_detection_tester.detection.stop()
            break
