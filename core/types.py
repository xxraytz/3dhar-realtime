import logging
import math
import time
from typing import NamedTuple

from multiprocessing import Queue, Process, Pipe, Event

from pykinect2_emu.pykinect2_emu import KinectEmulator
from processes.id_monitoring import MState
from processes.fall_event import send_preprocessing, Call


class SJoints(NamedTuple):
    """Coordinates for paint skeleton joints on a screen"""
    x: int = 0
    y: int = 0


class Joint(NamedTuple):
    """Skeleton coordinate with statements of confidence"""
    x: float = 0.
    y: float = 0.
    z: float = 0.
    state: int = 0

    def tolist(self):
        return [
            self.x, self.y, self.z, float(self.state)
        ]


class SkeletonFrame:

    def __init__(self, coords, screen_coords, player_id='NoID') -> object:
        """constructor"""
        self.player_coordinates = coords
        self.player_screen_coordinates = screen_coords
        self.player_ID = player_id

    def set_id(self, id):
        self.player_ID = id

    def is_zero_state(self):
        """Check defective skeleton"""
        if len(self.player_coordinates) == 0:
            return None
        for joint in self.player_coordinates:
            if joint.state == 0:
                return False
        return True


class KinectFrame:

    def __init__(self, max_body_count, color_frame=None):
        self.color_frame = color_frame
        self.depth_frame = []
        self.skeleton_frames = [None] * max_body_count
        self.__depth_frame_height = 0
        self.__depth_frame_width = 0
        self.__color_frame_height = 0
        self.__color_frame_width = 0

        self.__path = ""

    def reset(self, max_body_count):
        self.skeleton_frames = [None] * max_body_count

    def get_color_width(self):
        return self.__color_frame_width

    def get_color_height(self):
        return self.__color_frame_height

    def get_depth_width(self):
        return self.__depth_frame_width

    def get_depth_height(self):
        return self.__depth_frame_height

    def set_color_frame(self, color_frame, size):
        self.__color_frame_width, self.__color_frame_height = size
        self.color_frame = color_frame

    def set_color_width(self, value):
        self.__color_frame_width = value

    def set_color_height(self, value):
        self.__color_frame_height = value

    def set_depth_width(self, value):
        self.__depth_frame_width = value

    def set_depth_height(self, value):
        self.__depth_frame_height = value

    def is_empty(self):
        return len(self.skeleton_frames) == 0
    # def save_fall_statements(self):
    #     filename = self.__path + "falls.txt"
    #     with open(filename, "a+", encoding="utf-8") as f:
    #         f.write("Fall_start " + str(self.start_fall) + ", Fall_end: " + str(self.end_fall) + "\n")
    #         self.start_fall = -1
    #         self.end_fall = -1


class KinectFrameReader:
    def __init__(self, kinect: KinectEmulator):
        self.kinect = kinect
        self.bodies = None
        self.__has_body_tracked = False
        self.kinect_joints = None
        self.joints = None

    def get_max_body_count(self):
        return self.kinect.max_body_count

    def get_color_size(self):
        return self.kinect.color_frame_desc.Width, self.kinect.color_frame_desc.Height

    def has_color_frame(self):
        return self.kinect.has_new_color_frame()

    def has_new_bodies(self):
        return self.kinect.has_new_body_frame()

    def get_body_frame(self):
        return self.kinect.get_last_body_frame()

    def get_color_frame(self):
        if self.has_color_frame():
            return self.kinect.get_last_color_frame()
        else:
            return None

    def set_bodies(self):
        if self.has_new_bodies():
            bodies = self.get_body_frame()
            self.bodies = bodies.bodies
            self.__has_body_tracked = False
            return True

    def get_bodies(self):
        return self.bodies

    def get_body(self, index):
        return self.bodies[index]

    def is_body_tracked(self, index):
        if self.bodies[index].is_tracked:
            self.__has_body_tracked = True
            return True
        else:
            return False

    def has_body_tracked(self):
        return self.__has_body_tracked

    def get_joints(self, index):
        kinect_joints = self.bodies[index].joints
        # for joint in body.joints:
        #     joints.append([joint.Position.x * 1e3,
        #                    joint.Position.y * 1e3,
        #                    joint.Position.z * 1e3,
        #                    joint.TrackingState])

        return kinect_joints

    def get_skeleton_space_joints(self, index, usable_joints):
        joints = []
        kinect_joints = self.bodies[index].joints
        for j in usable_joints:
            joints.append([kinect_joints[j].Position.x * 1e3,
                           kinect_joints[j].Position.y * 1e3,
                           kinect_joints[j].Position.z * 1e3,
                           kinect_joints[j].TrackingState])
        # for joint in body.joints:
        #     joints.append([joint.Position.x * 1e3,
        #                    joint.Position.y * 1e3,
        #                    joint.Position.z * 1e3,
        #                    joint.TrackingState])
        return joints

    def get_color_space_joints(self, index, usable_indexes):
        joints = []
        for joint in self.kinect.body_joints_to_color_space(self.get_body(index).joints):
            if math.isinf(joint.x) and math.isinf(joint.y):
                joints.append([0, 0])
            else:
                joints.append([int(joint.x), int(joint.y)])
        usable_joints = []
        for j in usable_indexes:
            usable_joints.append(joints[j])
        return usable_joints


class PersonIdentificationOld:
    def __init__(self, id_delay, t_monitoring, bodies_count, conf):
        self.__bodies_count = bodies_count
        self.__conf = conf
        self.__t = t_monitoring
        self.__after_monitoring_delay = id_delay

        self.__result_obtained_time = -1.
        self.__start_time = 0.

        self.__people_indexes = ['NoID'] * bodies_count
        self.__parent_conn, self.__child_conn = Pipe()
        self.__process_person_identification = None
        self.__people_names = None

        # Bool constant - conditions
        self.__monitoring_completed = False
        self.__result_waiting = False
        self.__result_obtained = False
        self.__monitoring_process = True

        self.event = None

    def is_monitoring_ready(self):
        if self.is_monitoring_completed():
            return time.time() - self.__result_obtained_time > self.__after_monitoring_delay
        else:
            return False

    def is_monitoring(self):
        return self.__monitoring_process

    def is_monitoring_completed(self):
        return self.__monitoring_completed

    def is_monitoring_time_expired(self):
        if time.time() - self.__start_time > self.__t:
            return True
        else:
            return False

    def is_result_waiting(self):
        return self.__result_waiting

    def is_result_obtained(self):
        if self.__parent_conn.poll(timeout=0.01):
            self.__result_waiting = False
            self.__result_obtained = True
        return self.__result_obtained

    def append_frame(self, frame):
        self.__parent_conn.send((MState.APPEND_MONITORING, frame))

    def stop_monitoring(self):
        self.__monitoring_process = False
        self.__parent_conn.send(MState.GET_PERSON_ID)
        self.__result_waiting = True
        self.__people_indexes = ["Reloading..."] * self.__bodies_count

    def start_monitoring(self):
        self.__start_time = time.time()
        self.__result_obtained_time = -1
        self.__monitoring_process = True
        self.__monitoring_completed = False
        self.__result_waiting = False
        self.__result_obtained = False
        self.event = Event()

        self.__process_person_identification = Process(target=identify, args=(self.__child_conn,
                                                                              self.event))
        self.__process_person_identification.start()

        self.__parent_conn.send(MState.CHECK_STATUS)
        self.__parent_conn.send((MState.INIT_MONITORING, self.__bodies_count))

    def request_all_people_names(self):
        self.__parent_conn.send((MState.GET_PEOPLE,))
        self.event.wait()
        if self.__parent_conn.poll(timeout=1):
            self.__people_names = self.__parent_conn.recv()
            return self.__people_names
        else:
            return None

    def get_body_id(self, index):
        return self.__people_indexes[index]

    def get_all_people_names(self):
        return self.__people_names

    def get_people_ids(self):
        people_ids = ['NoID'] * self.__bodies_count
        ids = self.__parent_conn.recv()
        for j, person_id in enumerate(ids):
            people_ids[j] = person_id
        self.__process_person_identification.join()
        self.__process_person_identification.terminate()
        self.__people_indexes = people_ids
        self.__result_obtained_time = time.time()
        self.__monitoring_completed = True
        return people_ids

    def exit(self):
        self.__parent_conn.send(MState.END_MONITORING)
        self.__process_person_identification.join()


class VideoRecorder:
    def __init__(self, conf):
        self.__queue_video_recording = Queue()
        self.__process_video_recording = Process(target=send_preprocessing, args=(self.__queue_video_recording, conf))

    def start_recording(self):
        self.__process_video_recording.start()

    def open_recording_stream(self, store_dir, color_size):
        self.__queue_video_recording.put((Call.VIDEO_OPEN_STREAMING, [
            store_dir,
            color_size
        ]))

    def close_recording_stream(self):
        self.__queue_video_recording.put(Call.VIDEO_CLOSE_STREAMING)
        self.__process_video_recording.join()


class VideoStreaming:
    def __init__(self, conf, paint):
        self.__parent_conn, self.__child_conn = Pipe()
        self.__process_video_streaming = Process(target=paint, args=(self.__child_conn,
                                                                     conf))
        self.started = False

    def is_video_started(self):
        return

    def start_streaming(self):
        self.__process_video_streaming.start()
        self.started = True

    def is_there_image_to_print(self):
        return self.__parent_conn.poll(timeout=None)

    def get_frame(self, emitter):
        emitter(self.__parent_conn.recv())

    def put_frame(self, frame):
        self.__parent_conn.send(frame)

    def is_streaming_stopped(self):
        if self.started:
            return self.__process_video_streaming.exitcode is not None
        else:
            return False

    def clear_streaming(self):
        self.__parent_conn.send(None)

    def close_streaming(self):
        self.__process_video_streaming.join()
