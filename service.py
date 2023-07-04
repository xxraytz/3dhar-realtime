import argparse
import json
import logging
import os
from datetime import datetime

import cv2
import numpy as np
import time
from enum import Enum

from pykinect2_emu.pykinect2_emu import KinectEmulator

import comtypes
from pykinect2 import PyKinectV2, PyKinectRuntime

# from core import types
from core.types import SkeletonFrame, VideoRecorder, VideoStreaming
from core.types import KinectFrame
from core.types import KinectFrameReader
from core.types import PersonIdentification
from core.types import Joint, SJoints
from processes.paint import paint

import sys
from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton
from PyQt6.QtGui import QPixmap, QColor, QImage, QCloseEvent
from PyQt6.QtCore import Qt, QThread, pyqtSlot, pyqtSignal


# CONSTANTS
KINECT_FREQUENCY = 0.033
SKELETON_POINTS_COUNT: int
SKELETON_COLORS = []
USABLE_SK_JOINTS = {}
SKELETON_LINES = []
EXCLUDE_SKELETON_DISTANCES = []
SKELETON_HEIGHTS_CORRECTION: int


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


def get_config(args):
    global EXCLUDE_SKELETON_DISTANCES, SKELETON_COLORS, USABLE_SK_JOINTS, SKELETON_LINES, SKELETON_POINTS_COUNT, SKELETON_HEIGHTS_CORRECTION
    with open(args.config, 'r') as stream:
        config = json.load(stream)
        EXCLUDE_SKELETON_DISTANCES = config["EXCLUDE_SKELETON_DISTANCES"]
        SKELETON_COLORS = config["SKELETON_COLORS"]
        USABLE_SK_JOINTS = config["SKELETON_USABLE_JOINTS_NUMBER"]
        USABLE_SK_JOINTS = dict(
            sorted(
                USABLE_SK_JOINTS.items(),
                key=lambda x: x[1]
            )
        )
        SKELETON_LINES = config["SKELETON_LINES"]
        SKELETON_POINTS_COUNT = config["SKELETON_POINTS_COUNT"]
        SKELETON_HEIGHTS_CORRECTION = config["SKELETON_HEIGHTS_CORRECTION"]
    config = args.config
    return config


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='JSON config')
    parser.add_argument('-v', default=1, help='Kinect data visualisation')
    return parser.parse_args()


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


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._pause_flag = False

    def run(self):
        self.human_reco_service()
        # cap = cv2.VideoCapture(0)
        # while self._run_flag:
        #     if not self._pause_flag:
        #         ret, cv_image = cap.read()
        #         if ret:
        #             self.change_pixmap_signal.emit(cv_image)  # type: ignore
        # cap.release()

    def human_reco_service(self):
        args = parse_arguments()

        logger.info(f'Load config from {args.config}')
        logger.info(f'Starting service...')
        logger.info(f'Connecting to Microsoft Kinect v2')

        config = get_config(args)
        visual = args.v

        video_recorder = VideoRecorder(config)
        video_recorder.start_recording()

        person_identification = PersonIdentification(id_delay=5, t_monitoring=10, bodies_count=6, conf=config)

        # capture_picture()

        with KinectEmulator(os.environ.get('emulator_base_path')) as kinect:
            #
            # with PyKinectRuntime.PyKinectRuntime(
            #         PyKinectV2.FrameSourceTypes_Color
            #         | PyKinectV2.FrameSourceTypes_Body
            #         | PyKinectV2.FrameSourceTypes_Depth) as kinect:

            kinect_device = KinectFrameReader(kinect)

            if visual == VisualMode.VIDEO.value:
                video_streaming = VideoStreaming(config, paint)
                video_streaming.start_streaming()

            time.sleep(2)

            video_recorder.open_recording_stream(create_store_directory(f'recs/{os.environ.get("test_person_name")}/'),
                                                 kinect_device.get_color_size())

            person_identification.start_id()

            while self._run_flag:
                time.sleep(KINECT_FREQUENCY)
                if not person_identification.is_monitoring_blocked():
                    person_identification.start_id()

                kinect_frame = KinectFrame(kinect_device.get_max_body_count())
                kinect_frame.set_color_frame(kinect_device.get_color_frame())

                if kinect_device.set_bodies():
                    for body_index in range(0, kinect_device.get_max_body_count()):
                        if not kinect_device.is_body_tracked(body_index):
                            continue

                        joints_skeleton_space = kinect_device.get_joints(body_index)  # Joints in a skeleton space
                        joints_color_space = kinect_device.get_color_space_joints(body_index)

                        player_coordinates = [Joint(*joints_skeleton_space[j]) for j in USABLE_SK_JOINTS.values()]

                        player_screen_coordinates = [SJoints(*joints_color_space[j]) for j in USABLE_SK_JOINTS.values()]

                        kinect_frame.skeleton_frames[body_index] = SkeletonFrame(player_coordinates,
                                                                                 player_screen_coordinates,
                                                                                 person_identification.get_body_id(
                                                                                     body_index))

                    if visual == VisualMode.VIDEO.value:
                        video_streaming.put_frame({body_index: kinect_frame})
                        if video_streaming.is_there_image_to_print():
                            cimage = video_streaming.get_frame()
                            print(cimage.shape)
                            self.change_pixmap_signal.emit(cimage)  # type: ignore

                    if person_identification.is_monitoring():
                        person_identification.append_frame(kinect_frame.skeleton_frames)

                    if person_identification.is_monitoring_time_passed():
                        person_identification.stop_monitoring()

                    if person_identification.is_identification_delay_passed():
                        person_identification.get_people_ids()

                    if video_streaming.is_streaming_stopped():
                        video_streaming.clear_streaming()
                        video_recorder.close_recording_stream()
                        person_identification.exit()
                        video_streaming.close_streaming()
                        self._run_flag = False

        # Close Kinect sensor, close the window and quit.
        logger.info(f'Microsoft Kinect v2 was disconnected')
        logger.info(f'Stopping service...')
        logger.info(f'Service stopped.')

    def stop(self):
        self._run_flag = False
        self.wait()

    def pause(self):
        self._pause_flag = not self._pause_flag


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV video demo")
        self.size = (640, 480)
        self.image_label = QLabel(self)
        self.image_label.resize(*self.size)
        self.text_label = QLabel('OpenCV demo')
        self.pause_button = QPushButton()
        self.pause_button.setText('pause')
        self.pause_button.setFixedWidth(300)
        self.pause_button.clicked.connect(self.pause)

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.text_label)
        vbox.addWidget(self.pause_button)
        self.setLayout(vbox, )
        grey_image = QPixmap(*self.size)
        grey_image.fill(QColor('grey'))
        self.image_label.setPixmap(grey_image)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)  # type: ignore
        self.thread.start()

    def pause(self):
        self.thread.pause()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.thread.stop()
        event.accept()

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        converted_image = QImage(rgb_image, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled_converted_image = converted_image.scaled(*self.size, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(scaled_converted_image)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_image):
        qt_image = self.convert_cv_qt(cv_image)
        self.image_label.setPixmap(qt_image)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec())