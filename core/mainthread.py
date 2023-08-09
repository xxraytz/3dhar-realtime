import argparse
import json
import logging
import os
import queue
from datetime import datetime
import time
from enum import Enum

from pykinect2_emu.pykinect2_emu import KinectEmulator

import comtypes
from pykinect2 import PyKinectV2, PyKinectRuntime

import cv2
import numpy as np

# from core import types
from core.types import SkeletonFrame, VideoRecorder, VideoStreaming
from core.types import KinectFrame
from core.types import KinectFrameReader
from core.types import Joint, SJoints
from processes.paint import paint, VideoMode
from processes.id_monitoring import PersonIdentification

from PyQt6.QtCore import Qt, QThread, pyqtSlot, pyqtSignal, QElapsedTimer, QTimer

# CONSTANTS
KINECT_FREQUENCY = 0.04


# SKELETON_POINTS_COUNT: int
# SKELETON_COLORS = []
# USABLE_SK_JOINTS = {}
# SKELETON_LINES = []
# EXCLUDE_SKELETON_DISTANCES = []
# SKELETON_HEIGHTS_CORRECTION: int


class VisualMode(Enum):
    VIDEO = '1'
    NoVideo = '0'


def capture_picture():
    logger.info('Kinect color image saving...')
    with PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color
            | PyKinectV2.FrameSourceTypes_Body
            | PyKinectV2.FrameSourceTypes_Depth) as kinect:
        time.sleep(2)
        while True:
            if kinect.has_new_color_frame():
                color_image = kinect.get_last_color_frame()
                color_image = np.reshape(color_image,
                                         (kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4))
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2RGBA)
                cv2.imwrite('_color_image.png', color_image)
                break
    logger.info('Kinect color image saved.')
    exit(0)


def create_store_directory(file):
    if not os.path.isdir(file):
        os.mkdir(file)
    path = file + f'Scene_{datetime.now().date()}_' \
                  f'{datetime.now().time().hour}-' \
                  f'{datetime.now().time().minute}-' \
                  f'{datetime.now().time().second}' + "/"
    os.mkdir(path)
    os.mkdir(path + "Body/")
    os.mkdir(path + "Color/")
    os.mkdir(path + "Depth/")
    return path


logging.getLogger(comtypes.logger.name).setLevel(logging.CRITICAL)
logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VideoStreamingThread(QThread):
    video_frame_sending = pyqtSignal(np.ndarray)

    def __init__(self, config):
        super().__init__()
        self.queue = queue.Queue()
        self._run_flag = True
        self._pause_flag = False
        self.__video_mode = VideoMode.SKELETAL
        self.conf = config
        self.__indexes = ['NoID'] * 6

    def run(self):
        while True:
            frame = self.queue.get()
            if frame:
                image = paint(frame, self.conf, self.__video_mode, self.__indexes)
                self.video_frame_sending.emit(image)
            else:
                continue

    def stop(self):
        self._run_flag = False
        self.wait()

    def set_real_video_mode(self):
        self.__video_mode = VideoMode.REAL

    def set_skeletal_video_mode(self):
        self.__video_mode = VideoMode.SKELETAL

    def set_indexes(self, inds):
        self.__indexes = inds

    def pause(self):
        self._pause_flag = not self._pause_flag


class PeopleMonitoringThread(QThread):
    send_calculated_indexes = pyqtSignal(list)
    send_debug_info = pyqtSignal(dict, dict)
    send_people_information = pyqtSignal(list)

    def __init__(self, config, body_number):
        super().__init__()
        self._run_flag = True
        self._pause_flag = False
        self.queue = queue.Queue()
        self.__t = 10 * 1e3  # sec
        self.__delay = 5 * 1e3  # sec
        self._busy_flag = False
        self.pid = PersonIdentification(body_number)
        self.__people = None

    def database_connect(self):
        self.pid.database_connect()
        self.__people = self.pid.get_people()
        self.send_people_information.emit(self.__people)

    def run(self):
        self.pid.start_monitoring()
        timer = QElapsedTimer()
        timer.start()
        while timer.elapsed() < self.__t:
            frame = (self.queue.get()).skeleton_frames
            if frame:
                self.pid.append_frame(frame)
        self._busy_flag = True
        pids, debug_info, debug_one_class = self.pid.get_person_id(self.__people)
        logger.info(pids)
        self.send_calculated_indexes.emit(pids)
        self.send_debug_info.emit(debug_info, debug_one_class)
        self._busy_flag = False

    def is_busy(self):
        return self._busy_flag

    def set_monitoring_people(self, people):
        self.__people = people

    def stop(self):
        QTimer.singleShot(self.__delay, self.start)
        self._run_flag = False
        self.wait()

    def get_all_people(self):
        return self.pid.get_people()

    def pause(self):
        self._pause_flag = not self._pause_flag


class MainThread(QThread):
    video_frame_generate = pyqtSignal(KinectFrame)

    def __init__(self, config):
        super().__init__()
        self._run_flag = True
        self._pause_flag = False
        with open(config, 'r') as stream:
            config = json.load(stream)
            self.using_skeleton_joints = config["SKELETON_USABLE_JOINTS_NUMBER"]
            self.using_skeleton_joints = dict(
                sorted(
                    self.using_skeleton_joints.items(),
                    key=lambda x: x[1]
                )
            )

    def run(self):
        self.human_reco_service()

    def human_reco_service(self):

        # capture_picture()

        # with KinectEmulator(os.environ.get('emulator_base_path')) as kinect:

        with PyKinectRuntime.PyKinectRuntime(
                PyKinectV2.FrameSourceTypes_Color
                | PyKinectV2.FrameSourceTypes_Body
                | PyKinectV2.FrameSourceTypes_Depth) as kinect:
            #
            kinect_device = KinectFrameReader(kinect)
            time.sleep(1)

            while self._run_flag:
                if not self._pause_flag:
                    time.sleep(KINECT_FREQUENCY)  # KINECT_FREQUENCY

                    kinect_frame = KinectFrame(kinect_device.get_max_body_count())
                    kinect_frame.set_color_frame(kinect_device.get_color_frame(), kinect_device.get_color_size())

                    if kinect_device.set_bodies():
                        for body_index in range(0, kinect_device.get_max_body_count()):
                            if not kinect_device.is_body_tracked(body_index):
                                continue
                            _ = kinect_device.get_joints(body_index)  # Joints in a skeleton space
                            joints_skeleton_space = kinect_device.get_skeleton_space_joints(body_index, self.using_skeleton_joints.values())  # Joints in a skeleton space

                            joints_color_space = kinect_device.get_color_space_joints(body_index, self.using_skeleton_joints.values())

                            player_coordinates = [Joint(*j) for j in
                                                  joints_skeleton_space]

                            player_screen_coordinates = [SJoints(*j) for j in joints_color_space]

                            kinect_frame.skeleton_frames[body_index] = SkeletonFrame(player_coordinates,
                                                                                     player_screen_coordinates)

                        self.video_frame_generate.emit(kinect_frame)
                else:
                    time.sleep(0.5)

    def stop(self):
        self._run_flag = False
        self.wait()

    def pause(self):
        self._pause_flag = not self._pause_flag
