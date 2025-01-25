import os

import cv2 as cv2
import numpy as np


""" Kinect v2 emulator"""


class ColorFrameDescription:
    def __init__(self, width=0, height=0):
        self.Width = width
        self.Height = height


class Position:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class ScreenJoint:

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Joint:

    def __init__(self, x, y, z, state):
        self.Position = Position(x / 1000, y / 1000, z / 1000)
        self.TrackingState = int(state)

    def __str__(self):
        return f'{self.Position.x} {self.Position.y} {self.Position.z} {self.TrackingState}'


class Bodies:
    def __init__(self, bodies):
        self.bodies = bodies


class Body:
    def __init__(self, joints, is_tracked):
        self.joints = joints
        self.is_tracked = is_tracked


class KinectEmulator:

    KINECT_COLOR_WIDTH = 1920
    KINECT_COLOR_HEIGHT = 1080

    def __init__(self, path):
        self.__path_to_data = path if path is not None else "data/.."
        self.__body_data = self.__path_to_data + "/Body/FileSkeleton.txt"
        self.__color_data = self.__path_to_data + '/Color/person_stream.mp4'
        self.__frame_number = -1
        self.color_frame_desc = ColorFrameDescription(width=self.KINECT_COLOR_WIDTH,
                                                      height=self.KINECT_COLOR_HEIGHT)
        self.max_body_count = 6
        self.__color_coordinates = []
        self.__exceed_points = [7, 11, 15, 19, 21, 22, 23, 24]
        self.__color_skeleton_frames = self.__get_color_body_frames()
        self.__skeleton_frames = self.__get_body_frames()
        self.__color_frames_gen = self.__color_frames_generator()
        self.__current_color_frame = None

    def __get_body_frames(self):
        with open(self.__body_data, 'r') as file:
            skeleton_frames = []
            for line in file:
                if 'frame' in line.lower():
                    skeleton = []
                    for i in range(17):
                        skeleton.append(
                            Joint(*[
                                float(el) for el in file.readline().split(' ')[:4]
                            ])
                        )
                    for i in self.__exceed_points:
                        skeleton.insert(i, Joint(0, 0, 0, 0))
                    skeleton_frames.append(skeleton)
        return skeleton_frames

    def __get_color_body_frames(self):
        with open(self.__body_data, 'r') as file:
            skeleton_frames = []
            for line in file:
                if 'frame' in line.lower():
                    skeleton = []
                    for i in range(17):
                        skeleton.append(
                            ScreenJoint(*[
                                float(el) for el in file.readline().split(' ')[4:]
                            ])
                        )
                    for i in self.__exceed_points:
                        skeleton.insert(i, ScreenJoint(0, 0))
                    skeleton_frames.append(skeleton)
        return skeleton_frames

    def __body_frames_generator(self):
        while True:
            yield [
                Body(self.__skeleton_frames[self.__frame_number], True)
            ] + [
                Body(self.__skeleton_frames[self.__frame_number], False)
            ] * (self.max_body_count - 1)

    def __body_color_frames_generator(self):
        while True:
            yield [
                self.__color_skeleton_frames[self.__frame_number]
            ] * self.max_body_count

    def __color_frames_generator(self):
        cap = cv2.VideoCapture(self.__color_data)
        while True:
            ret, frame = cap.read()
            if ret:
                self.__frame_number += 1
                yield frame
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print('Camera to zero')
                self.__frame_number = -1
                continue
        cap.release()

    @staticmethod
    def has_new_body_frame():
        return True

    def get_last_body_frame(self):
        return Bodies(next(self.__body_frames_generator()))

    def get_last_color_frame(self):
        return next(self.__color_frames_gen)

    @staticmethod
    def has_new_color_frame():
        return True

    def body_joints_to_color_space(self, joints):
        index = None
        color_frames = next(self.__body_color_frames_generator())
        for i, jnts in enumerate(next(self.__body_frames_generator())):
            if [joints[j].__str__() == jnts.joints[j].__str__() for j in range(len(joints))] == [True] * len(joints):
                index = i
        return color_frames[index]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
