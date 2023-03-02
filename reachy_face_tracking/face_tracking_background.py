"""."""
import logging
import numpy as np
import time

from reachy_sdk import ReachySDK
from .head_controller import HeadController
from .detection import Detection

from collections import deque
from pathlib import Path

logger = logging.getLogger('reachy.face.tracking')

model_path = '/home/reachy/dev/reachy-face-tracking/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite'


class FaceTrackingBackground:
    """."""
    def __init__(self, host: str = 'localhost'):
        self.reachy = ReachySDK(host=host)

        self.prev_y, self.prev_z = 0, 0
        self.cmd_y, self.cmd_z = self.prev_y, self.prev_z
        self.xM = 0
        self.yM = 0
        self.target_size = 0
        self.center = np.array([0, 0])

        self.queue = deque([], 50)

        self.detection = Detection(self.reachy, detection_model_path=model_path)
        self.controller = HeadController([0, 0], cb=self.servoing, pid_params=[0.0004, 0.0001, 0, 0, 0.017, 0.002])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        logger.info(
            'Closing the playground',
            extra={
                'exc': exc,
            }
        )
        self.reachy.turn_off('head')

    async def setup(self):  # MODIFIED
        logger.info('Setup Reachy before starting.')
        self.reachy.turn_on('head')
        self.reachy.head.look_at(0.5, 0, 0, 1.5)

        self.center = np.array([160, 160])
        self.queue.append(self.prev_y)

    # Functions related to tracking

    def servoing(self, res):
        x = 0.5
        y, z = res

        try:
            thetas = self.reachy.head._look_at(x, y, z)
            self.reachy.head.neck_roll.goal_position = thetas[self.reachy.head.neck_roll]
            self.reachy.head.neck_pitch.goal_position = thetas[self.reachy.head.neck_pitch]
            self.reachy.head.neck_yaw.goal_position = thetas[self.reachy.head.neck_yaw]

        except ValueError:
            return

    def activate_tracking_mode(self):
        self.controller.start()

    def deactivate_tracking_mode(self):
        self.controller.stop()

    def get_target_info(self):
        self.xM, self.yM, self.target_size = self.detection._face_target
        if len(self.queue) == 0:
            self.queue.append(self.prev_y)

    def track(self):
        self.cmd_y, self.cmd_z = self.controller.track(
            [self.cmd_y, self.cmd_z], [self.prev_y, self.prev_z],
            goal=self.center,
            input_controller=[self.xM, self.yM]
            )
        self.prev_y, self.prev_z = self.cmd_y, self.cmd_z
        self.queue.append(self.prev_y)

    def look_at_previous_target(self):
        self.controller.set_new_target([self.prev_y, self.prev_z])

    def reinitialize_target(self):
        self.prev_y, self.prev_z = self.reachy.head._previous_look_at[1], self.reachy.head._previous_look_at[2]
        self.cmd_y, self.cmd_z = self.prev_y, self.prev_z
        self.controller.origin = np.array([self.reachy.head._previous_look_at[1], self.reachy.head._previous_look_at[2]])
        self.controller.target = np.array([self.reachy.head._previous_look_at[1], self.reachy.head._previous_look_at[2]])
        self.controller.t0 = time.time()
        self.controller.last_update.clear()
        self.detection._prev_face_target = [0, 0, 0]
