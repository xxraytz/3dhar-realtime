import json
from core.types import KinectFrame
import cv2 as cv2
import numpy as np
from enum import Enum


class VideoMode(Enum):
    REAL = 49
    SKELETAL = 50

    def get_all(self):
        return [self.REAL.value, self.SKELETAL.value]


def paint(q_paint, config):
    skeleton_rgb = (0,0,255)
    video_mode = VideoMode.SKELETAL
    saved_color_image = cv2.imread('color_image1.png')
    window_name = 'Fall Detection System'
    fall_detection_text = 'Fall Detection'

    with open(config, 'r') as stream:
        config = json.load(stream)
        skeleton_lines = config["SKELETON_LINES"]

    while True:
        paint_frame = q_paint.get()
        if paint_frame is None:
            print('break')
            break
        body_number = list(paint_frame.keys())[0]
        frame: KinectFrame = paint_frame.get(body_number)
        skeleton = frame.player_screen_coordinates
        if video_mode is VideoMode.REAL:
            color_image = frame.color_frame
            color_image = np.reshape(color_image, (frame.get_color_height(), frame.get_color_width(), 4))
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2RGBA)
        elif video_mode is VideoMode.SKELETAL:
            color_image = saved_color_image.copy()
        else:
            color_image = None
        if len(skeleton) != 0:
            for line in skeleton_lines:
                cv2.line(color_image,
                         skeleton[0][line[0]], skeleton[0][line[1]], skeleton_rgb, 5)
        color_image = cv2.resize(color_image, (1280, 800))

        cv2.imshow(window_name, color_image)
        key = cv2.waitKey(1)
        if key in video_mode.get_all():
            video_mode = VideoMode(key)
        if key == ord('q'):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) == 0:
            break

    cv2.destroyAllWindows()