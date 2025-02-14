import cv2 as cv2
import mediapipe as mp
import numpy as np
from tools.GoogleSkeletonUnification import convert_to_kinect_coords

"""Camera Streamer"""


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
        #self.Position = Position(x / 1000, y / 1000, z / 1000)
        self.Position = Position(x, y, z)
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


class CameraStreamer:
    CAMERA_STREAM_WIDTH = 1920
    CAMERA_STREAM_HEIGHT = 1080

    def __init__(self, path, pose_estimator: str = "Mediapipe"):
        self.estimator_name = pose_estimator
        self.__path_to_data = path if path is not None else "data/.."
        self.__body_data = None  # self.__path_to_data + "/Body/FileSkeleton.txt"
        self.__color_data = self.__path_to_data + '/Color/person_stream.mp4'
        self.__frame_number = -1
        self.color_frame_desc = ColorFrameDescription(width=self.CAMERA_STREAM_WIDTH,
                                                      height=self.CAMERA_STREAM_HEIGHT)
        self.max_body_count = 1  # 6
        self.__color_coordinates = []
        self.__exceed_points = [7, 11, 15, 19, 21, 22, 23, 24]
        self.__color_frames_gen = self.__color_frames_generator()

        self.__pose_estimator = self.__set_pose_estimator()
        self.__current_color_frame = None

        self.__unify = convert_to_kinect_coords

    def __set_pose_estimator(self):
        if self.estimator_name == 'Mediapipe':
            mp_pose = mp.solutions.pose
            pe = mp_pose.Pose()
            return pe.process

    def __body_frames_generator(self):
        """
        Генератор возвращает скелетную модель в пространстве сенсора. Три координаты
        x, y, z
        и confidence coefficient: state
        """
        while True:
            frame_skeleton = self.__pose_estimator(self.__current_color_frame)
            if frame_skeleton.pose_world_landmarks is None:
                yield None
            else:
                frame_skeleton = self.__unify(frame_skeleton.pose_world_landmarks.landmark, True)

            skeleton = []
            for sk in frame_skeleton:
                skeleton.append(Joint(*sk, 2))

            for i in self.__exceed_points:
                skeleton.insert(i, Joint(0, 0, 0, 0))
            yield [
                Body(skeleton, True)
            ] + [
                Body(skeleton, False)
            ] * (self.max_body_count - 1)

    def __body_color_frames_generator(self):
        """
        Генератор возвращает скелетную модель в пространстве экрана.
        Две координаты
        x, y
        """
        while True:
            frame_skeleton = self.__pose_estimator(self.__current_color_frame)
            if frame_skeleton.pose_landmarks is None:
                yield None
            else:
                frame_skeleton = self.__unify(frame_skeleton.pose_landmarks.landmark, False)
            skeleton = []
            for sk in frame_skeleton:
                skeleton.append(ScreenJoint(*(sk[:2])))

            for i in self.__exceed_points:
                skeleton.insert(i, ScreenJoint(0, 0))
            yield [
                skeleton
            ] * self.max_body_count

    def __color_frames_generator(self):
        """
        Генератор вовзращает кадр с видеопотока.
        """
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                self.__frame_number += 1
                yield cv2.flip(frame, 1)
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
        """
        Функция возвращает следующую скелетную модель.
        """
        return Bodies(next(self.__body_frames_generator()))

    def get_last_color_frame(self):
        """
        Функция возвращает следующий кадр видеоряда.
        """
        self.__current_color_frame = next(self.__color_frames_gen)
        return self.__current_color_frame

    @staticmethod
    def has_new_color_frame():
        return True

    def body_joints_to_color_space(self, joints):
        """
        Функция заглушка. Делает вид, что переводит координаты из пространства в пространства.
        На самом деле просто возвращает результат другого генератора - чет типо того.
        """
        skeleton = []
        color_frames = next(self.__body_color_frames_generator())
        # Один скелет возможен в кадре, поэтому индекс [0]
        for i, joint in enumerate(color_frames[0]):
            skeleton.append(ScreenJoint(joint.x * CameraStreamer.CAMERA_STREAM_WIDTH, joint.y * CameraStreamer.CAMERA_STREAM_HEIGHT))
        return skeleton



    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
