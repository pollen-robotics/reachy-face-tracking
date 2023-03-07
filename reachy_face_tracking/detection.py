"""Module handling face detection."""
import time
from threading import Thread

import numpy as np
from PIL import Image
import cv2 as cv

from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import detect


class Detection(object):
    """Main class using Coral EdgeTPU for face detection."""

    def __init__(self, reachy, detection_model_path):
        """Init class."""
        self.reachy = reachy

        self.interpreter = make_interpreter(detection_model_path)
        self.interpreter.allocate_tensors()
        self._input_interpreter_size = common.input_size(self.interpreter)

        self._t = None
        self.running = False
        self._somebody_detected = False

        self._face_target = [0, 0, 0]  # xM, yM, size
        self._prev_face_target = self._face_target
        self._face_emb = [0, 0, 0, 0, 0]  # x1, y1, x2, y2, size
        self._face_image = None
        self._somebody_detected = False
        self._time = [time.time()]
        self._img_index = 0

    def start(self):
        """Start detect thread."""
        if self._t is not None:
            return

        self.running = True
        self._t = Thread(target=self.detection_thread)
        self._t.start()

        while not self._t.is_alive():
            time.sleep(0.01)

    def stop(self):
        """Stop detect thread."""
        self.running = False
        self._t = None

    def is_playing(self):
        """Check if detect thread is alive."""
        if self._t is None:
            return False
        return self._t.is_alive()

    def detection_thread(self):
        """Detect faces in Reachy's last frame and update face to track."""
        while self.running:
            self._time.append(time.time())
            self._image = self.reachy.right_camera.last_frame
            pil_image = Image.fromarray(
                self._image[:, :, ::-1]
            ).resize(self._input_interpreter_size, Image.ANTIALIAS)

            candidates = self.face_detect(pil_image)

            if not candidates:
                self._somebody_detected = False

            else:
                self._somebody_detected = True
                sizes, faces, face_emb, dist_face = [], [], [], []
                for candidate in candidates:
                    x1, y1, x2, y2 = candidate.bbox.xmin, candidate.bbox.ymin, candidate.bbox.xmax, candidate.bbox.ymax
                    sizes.append((x2 - x1) * (y2 - y1))
                    faces.append([int((x1 + x2) / 2), int((y1 + y2) / 2)])
                    dist_face.append(
                        np.square(
                            ((x1 + x2) / 2) - self._prev_face_target[0]
                        ) + np.square(
                            ((y1 + y2) / 2) - self._prev_face_target[1]
                        ))
                    face_emb.append([x1, y1, x2, y2])

                if max(sizes) < 2.5 * self._prev_face_target[2]:
                    self._face_emb[:4] = face_emb[np.argmin(dist_face)]
                    self._face_emb[4] = sizes[np.argmin(dist_face)]
                    self._face_target[0], self._face_target[1] = faces[np.argmin(dist_face)]
                    self._face_target[2] = sizes[np.argmin(dist_face)]

                else:
                    self._face_emb[:4] = face_emb[np.argmax(max(sizes))]
                    self._face_emb[4] = max(sizes)
                    self._face_target[0], self._face_target[1] = faces[np.argmax(max(sizes))]
                    self._face_target[2] = max(sizes)

                self._prev_face_target = self._face_target
                self._face_image = cv.resize(self._image, self._input_interpreter_size)[
                    self._face_emb[1]:self._face_emb[3],
                    self._face_emb[0]:self._face_emb[2]
                ]

            time.sleep(0.01)

    def face_detect(self, image: Image):
        """Return the face detection inference result for a given image."""
        image.resize(self._input_interpreter_size, Image.ANTIALIAS)
        common.set_input(self.interpreter, image)
        self.interpreter.invoke()
        candidates = detect.get_objects(
            interpreter=self.interpreter,
            score_threshold=0.5,
        )
        return candidates
