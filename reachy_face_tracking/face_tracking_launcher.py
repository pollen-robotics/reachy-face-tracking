"""
Face tracking application for Reachy 2021.

This package defines an autonomous mode where Reachy tracks the face
of the person closest to it.
"""
import logging
import time
import asyncio

import zzlog

from .face_tracking_background import FaceTrackingBackground

logger = logging.getLogger('reachy.face.tracking')


async def run_tracking_loop(face_tracking_background):

    tracking_threshold = 20 * 20
    track_count = 0

    # nobody_here = True

    face_tracking_background.detection.start()
    face_tracking_background.activate_tracking_mode()
    time.sleep(0.02)

    while True:
        if face_tracking_background.detection._somebody_detected:
            logger.info(
                'Someone has been detected',
                extra={
                    'x': face_tracking_background.detection._face_target[0],
                    'y': face_tracking_background.detection._face_target[1],
                },
            )

            face_tracking_background.get_target_info()
            face_tracking_background.track()
            time.sleep(0.01)
            # # if someone is close enough to begin an interaction
            if face_tracking_background.target_size > tracking_threshold:
                track_count = 0

                face_tracking_background.activate_tracking_mode()
                face_tracking_background.track()

        else:
            if track_count < 40:
                track_count += 1
            else:
                logger.info(
                    'No one detected, Reachy plays random behavior.',
                )
                track_count = 0
                # flyer_background.deactivate_tracking_mode()
                time.sleep(0.2)
                # flyer_background.idleForever.start()
                # was_in_idle_mode = True

        time.sleep(0.02)


if __name__ == '__main__':

    import argparse

    from datetime import datetime
    from glob import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file')
    args = parser.parse_args()

    if args.log_file is not None:
        n = len(glob(f'{args.log_file}*.log')) + 1

        now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
        args.log_file += f'-{n}-{now}.log'

    _ = zzlog.setup(
        logger_root='',
        filename=args.log_file,
    )

    logger.info(
        'Initializing face tracking.'
    )

    async def f():
        with FaceTrackingBackground() as face_tracking_background:
            await face_tracking_background.setup()
            await run_tracking_loop(face_tracking_background)

    asyncio.run(f())
