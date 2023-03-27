import json
from core.types import ScreenJoint
from core.types import KinectFrame
import cv2 as cv2
import numpy as np
from enum import Enum


class VideoMode(Enum):
    REAL = 49
    SKELETAL = 50

    def get_all(self):
        return [self.REAL.value, self.SKELETAL.value]


def get_person_id(skeleton, ):
    print(skeleton)


def paint(q_paint, config):
    video_mode = VideoMode.SKELETAL
    saved_color_image = cv2.imread('_color_image.png')
    window_name = 'Fall Detection System'
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
        skeletons = frame.skeleton_frames
        if video_mode is VideoMode.REAL:
            color_image = frame.color_frame
            color_image = np.reshape(color_image, (frame.get_color_height(), frame.get_color_width(), 3))
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2RGBA)
        elif video_mode is VideoMode.SKELETAL:
            color_image = saved_color_image.copy()
        else:
            color_image = None
        if len(skeletons) != 0:
            for i, sk in enumerate(skeletons):
                if sk is not None:
                    screen_coordinates = sk.player_screen_coordinates
                    # get_person_id(sk.player_coordinates)
                    person_id = sk.player_ID
                    if person_id != 'NoID':
                        id_rgb = (96, 230, 0)
                    else:
                        id_rgb = (237, 72, 48)
                    for line in skeleton_lines:
                        if not (screen_coordinates[line[0]] == ScreenJoint(0, 0)
                                or screen_coordinates[line[1]] == ScreenJoint(0, 0)):
                            cv2.line(color_image,
                                     screen_coordinates[line[0]], screen_coordinates[line[1]], (255, 150, 0), 9)
                        for joint in screen_coordinates:
                            if joint != ScreenJoint(0, 0):
                                cv2.circle(color_image, joint, 15, color=(255, 220, 190), thickness=-1)
                                cv2.circle(color_image, joint, 10, color=(255, 190, 140), thickness=-1)
                        cv2.putText(color_image, person_id + f" {i}", sk.player_screen_coordinates[3],
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, id_rgb, 3, 2)

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
