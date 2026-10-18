"""Qt workers for capture, rendering, activity recognition and optional identity."""
import logging
import queue
import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.config import asset_path, load_config
from core.types import Joint, SJoints, SkeletonFrame, KinectFrame, KinectFrameReader
from processes.har import HAR
from processes.id_monitoring import PersonIdentification
from processes.paint import paint, VideoMode
from pykinect2_emu.pykinect2_emu import KinectEmulator

logger = logging.getLogger(__name__)
KINECT_FREQUENCY = 0.04


class Worker(QThread):
    failed = pyqtSignal(str)

    def run(self):
        try:
            self.process()
        except Exception as exc:
            logger.exception('%s failed', type(self).__name__)
            self.failed.emit(f'{type(self).__name__}: {exc}')

    def stop(self):
        self.requestInterruption()
        self.wait()

    def receive(self, frames):
        while not self.isInterruptionRequested():
            try:
                return frames.get(timeout=0.1)
            except queue.Empty:
                pass
        return None


class VideoStreamingThread(Worker):
    video_frame_sending = pyqtSignal(np.ndarray)

    def __init__(self, config):
        super().__init__()
        self.queue = queue.Queue(maxsize=1)
        self.conf = config
        self.__video_mode = VideoMode.SKELETAL
        self.__indexes = ['NoID'] * 6
        self.__classes = ['no_action'] * 6

    def process(self):
        while not self.isInterruptionRequested():
            frame = self.receive(self.queue)
            if frame is not None:
                image = paint(frame, self.conf, self.__video_mode, self.__indexes, self.__classes)
                self.video_frame_sending.emit(image)

    def set_real_video_mode(self):
        self.__video_mode = VideoMode.REAL

    def set_skeletal_video_mode(self):
        self.__video_mode = VideoMode.SKELETAL

    def set_indexes(self, indexes):
        self.__indexes = indexes

    def set_classes(self, classes):
        self.__classes = classes


class HARThread(Worker):
    send_action_info = pyqtSignal(list)

    def __init__(self, config, body_number):
        super().__init__()
        self.q_skeletons = queue.Queue(maxsize=1)
        self.__har = HAR(body_number, dimension='3D')
        self._busy_flag = False

    def database_connect(self):
        self.__har.database_connect()

    def process(self):
        while not self.isInterruptionRequested():
            self.__har.start_monitoring()
            for _ in range(32):
                frame = self.receive(self.q_skeletons)
                if frame is None:
                    return
                self.__har.append_frame(frame.skeleton_frames)
            self._busy_flag = True
            self.send_action_info.emit(self.__har.get_person_class())
            # Preserve the original two-second interval without restarting QThreads.
            deadline = time.monotonic() + 2
            while not self.isInterruptionRequested() and time.monotonic() < deadline:
                self.msleep(50)
            while not self.q_skeletons.empty():
                self.q_skeletons.get_nowait()
            self._busy_flag = False

    def is_busy(self):
        return self._busy_flag


class PeopleMonitoringThread(Worker):
    send_calculated_indexes = pyqtSignal(list)
    send_debug_info = pyqtSignal(dict, dict)
    send_people_information = pyqtSignal(list)

    def __init__(self, config, body_number):
        super().__init__()
        self.queue = queue.Queue(maxsize=1)
        self.pid = PersonIdentification(body_number)
        self._busy_flag = False
        self.__people = None

    def database_connect(self):
        self.pid.database_connect()
        self.__people = self.pid.get_people()
        self.send_people_information.emit(self.__people)

    def process(self):
        while not self.isInterruptionRequested():
            self.pid.start_monitoring()
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                frame = self.receive(self.queue)
                if frame is None:
                    return
                self.pid.append_frame(frame.skeleton_frames)
            self._busy_flag = True
            ids, debug, outliers = self.pid.get_person_id(self.__people)
            self.send_calculated_indexes.emit(ids)
            self.send_debug_info.emit(debug, outliers)
            deadline = time.monotonic() + 5
            while not self.isInterruptionRequested() and time.monotonic() < deadline:
                self.msleep(50)
            while not self.queue.empty():
                self.queue.get_nowait()
            self._busy_flag = False

    def is_busy(self):
        return self._busy_flag

    def set_monitoring_people(self, people):
        self.__people = list(people)


class MainThread(Worker):
    video_frame_generate = pyqtSignal(KinectFrame)

    def __init__(self, config, source='kinect', recording=None):
        super().__init__()
        self._pause_flag = False
        self.source = source
        self.recording = recording
        config = load_config(config)
        self.using_skeleton_joints = sorted(config['SKELETON_USABLE_JOINTS_NUMBER'].values())

    def open_source(self):
        if self.source == 'recording':
            return KinectEmulator(str(self.recording or asset_path('emulator_base_path')))
        # Kinect is Windows-only; replay and hardware-independent tests do not import COM.
        from pykinect2 import PyKinectV2, PyKinectRuntime
        return PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

    def process(self):
        with self.open_source() as sensor:
            reader = KinectFrameReader(sensor)
            color = None
            while not self.isInterruptionRequested():
                self.msleep(40)
                if self._pause_flag:
                    continue
                if reader.has_color_frame():
                    color = reader.get_color_frame()
                if color is None or not reader.set_bodies():
                    continue
                frame = KinectFrame(reader.get_max_body_count())
                frame.set_color_frame(color, reader.get_color_size())
                for index in range(reader.get_max_body_count()):
                    if not reader.is_body_tracked(index):
                        continue
                    coords = reader.get_skeleton_space_joints(index, self.using_skeleton_joints)
                    screen = reader.get_color_space_joints(index, self.using_skeleton_joints)
                    skeleton = SkeletonFrame([Joint(*j) for j in coords], [SJoints(*j) for j in screen])
                    skeleton.tracking_id = getattr(reader.get_body(index), 'tracking_id', index)
                    frame.skeleton_frames[index] = skeleton
                self.video_frame_generate.emit(frame)

    def pause(self):
        self._pause_flag = not self._pause_flag
