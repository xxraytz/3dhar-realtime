import json

from core.types import SJoints
from core.types import KinectFrame
import cv2 as cv2
import numpy as np
from enum import Enum


class Call(Enum):
    INIT = "init_video_stream"
    PUT = "put_frame_into_video_stream"


class VideoMode(Enum):
    REAL = 49
    SKELETAL = 50

    def get_all(self):
        return [self.REAL.value, self.SKELETAL.value]


def get_person_id(skeleton, ):
    print(skeleton)


def generate_image(saved_color_image, frame, video_mode, skeleton_lines, indexes):
    skeletons = frame.skeleton_frames
    if video_mode is VideoMode.REAL:
        color_image = frame.color_frame
        if len(color_image.shape) > 2:
            color_image = np.reshape(color_image, (frame.get_color_height(), frame.get_color_width(), 3))
        else:
            color_image = np.reshape(color_image, (frame.get_color_height(), frame.get_color_width(), 4))
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2RGBA)
    elif video_mode is VideoMode.SKELETAL:
        color_image = saved_color_image.copy()
    else:
        color_image = None
    if len(skeletons) != 0:
        for i, sk in enumerate(skeletons):
            if sk is not None:
                screen_coordinates = sk.player_screen_coordinates
                person_id = indexes[i]
                if person_id == 'NoID':
                    id_rgb = (237, 72, 48)
                elif person_id == 'RELOADING...':
                    id_rgb = (34, 254, 254)
                else:
                    id_rgb = (96, 230, 0)
                for line in skeleton_lines:
                    if not (screen_coordinates[line[0]] == SJoints(0, 0)
                            or screen_coordinates[line[1]] == SJoints(0, 0)):
                        cv2.line(color_image,
                                 screen_coordinates[line[0]], screen_coordinates[line[1]], (255, 150, 0), 9)
                    for joint in screen_coordinates:
                        if joint != SJoints(0, 0):
                            cv2.circle(color_image, joint, 15, color=(255, 220, 190), thickness=-1)
                            cv2.circle(color_image, joint, 10, color=(255, 190, 140), thickness=-1)
                    cv2.putText(color_image, f"{i}. {person_id}",
                                (sk.player_screen_coordinates[3][0] + 20, sk.player_screen_coordinates[3][1] + 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, id_rgb, 3, 2)
    return color_image


def paint(frame, config, video_mode, indexes):
    saved_color_image = cv2.imread('_color_image.png')
    with open(config, 'r') as stream:
        config = json.load(stream)
        skeleton_lines = config["SKELETON_LINES"]
    color_image = generate_image(saved_color_image, frame, video_mode, skeleton_lines, indexes)
    color_image = cv2.resize(color_image, (1920, 1080))
    return color_image