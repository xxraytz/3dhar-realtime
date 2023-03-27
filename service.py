import argparse
import json
import logging
import math
import os

import cv2
import numpy as np
import time
from enum import Enum
from multiprocessing import Queue, Process

from pykinect2_emu.pykinect2_emu import KinectEmulator
from processes.paint import paint
from processes.fall_event import send_preprocessing
from processes.id_monitoring import identify, MState

import comtypes
from pykinect2 import PyKinectV2, PyKinectRuntime

from core import types
from core.types import SkeletonFrame
from core.types import SvmWrapper
from core.types import KinectFrame

from collections import deque

# CONSTANTS
KINECT_FREQUENCY = 0.033
SKELETON_POINTS_COUNT: int
SKELETON_COLORS = []
SKELETON_USABLE_JOINTS_NUMBER = {}
SKELETON_LINES = []
EXCLUDE_SKELETON_DISTANCES = []
SKELETON_HEIGHTS_CORRECTION: int
host_image = np.array([0, 0])

# Global variables
gl_distances = []
gl_distances_speed = []
gl_distances_acceleration = []
gl_heights = []
gl_heights_speed = []
gl_heights_acceleration = []

curr_frame_number = 0
prev_frame_number = 0
upper_bound = 3
lower_bound = -3
fall_start_frame = -1
fall_end_frame = -1
outlier = False
skeleton_frames = []
terminate = False
prev_fall_state = False


class Call(Enum):
    LOGIN = "login"
    TEST = "test"
    VIDEO_OPEN = "open_video"
    VIDEO_RECORDING = "recording_video"
    VIDEO_CLOSE = "close_video"
    EXIT = "exit"
    VIDEO_APPEND = "video_append"
    VIDEO_CUT_SEND = "video_cut&send"
    VIDEO_STREAMING = "video_streaming"
    VIDEO_OPEN_STREAMING = "video_streaming_open"
    VIDEO_CLOSE_STREAMING = "video_streaming_close"


def get_max_height(coordinates):
    max_height = coordinates[SKELETON_USABLE_JOINTS_NUMBER.get("Head")].y + SKELETON_HEIGHTS_CORRECTION

    for frame in skeleton_frames:
        h = frame.get_coordinates()[SKELETON_USABLE_JOINTS_NUMBER.get("Head")].y + SKELETON_HEIGHTS_CORRECTION
        if h > max_height:
            max_height = h

    return max_height


def generate_lib_svm_vector(height):
    """Generate svm vector by distances between joints"""
    current_distance_acceleration = gl_distances_acceleration[frame_count]
    current_height_acceleration = gl_heights_acceleration[frame_count]

    if current_height_acceleration is None or current_distance_acceleration is None:
        return None

    current_distances = gl_distances[frame_count]
    current_heights = gl_heights[frame_count]
    current_distances_speed = gl_distances_speed[frame_count]
    current_heights_speed = gl_heights_speed[frame_count]

    assert height != 0
    svm_node = []
    # Distances
    svm_node.extend([dist / height for dist in current_distances])
    svm_node.extend([cur_height / height for cur_height in current_heights])
    svm_node.extend([speed / height for speed in current_distances_speed])
    svm_node.extend([cur_height / height for cur_height in current_heights_speed])
    svm_node.extend([accel / height for accel in current_distance_acceleration])
    svm_node.extend([cur_height / height for cur_height in current_height_acceleration])

    return svm_node


def generate_distances_matrix(local_coordinates):
    distances = np.zeros(0)
    for f_i in range(len(local_coordinates)):
        for f_j in range(f_i + 1, len(local_coordinates)):
            if [f_i, f_j] not in EXCLUDE_SKELETON_DISTANCES:
                distances = np.append(distances,
                                      math.sqrt(math.pow(local_coordinates[f_i].x - local_coordinates[f_j].x, 2)
                                                + math.pow(local_coordinates[f_i].y - local_coordinates[f_j].y, 2)
                                                + math.pow(local_coordinates[f_i].z - local_coordinates[f_j].z, 2)))

    gl_distances.append(distances)


def generate_heights_matrix(local_coordinates):
    heights = np.zeros(0)
    for f_i in range(len(local_coordinates)):
        heights = np.append(heights, local_coordinates[f_i].y + SKELETON_HEIGHTS_CORRECTION)

    gl_heights.append(heights)


def generate_distances_speed_matrix():
    if frame_count < 1:
        gl_distances_speed.append(None)
        return

    vec_prev_distances = gl_distances[frame_count - 1]
    vec_curr_distances = gl_distances[frame_count]
    distances_speed = vec_curr_distances - vec_prev_distances
    gl_distances_speed.append(distances_speed)


def generate_heights_speed_matrix():
    if frame_count < 1:
        gl_heights_speed.append(None)
        return

    vec_prev_height = gl_heights[frame_count - 1]
    vec_cur_height = gl_heights[frame_count]
    heights_speed = vec_cur_height - vec_prev_height
    gl_heights_speed.append(heights_speed)


def generate_distances_acceleration_matrix():
    if frame_count < 2:
        gl_distances_acceleration.append(None)
        return

    vec_prev_distances_speed = gl_distances_speed[frame_count - 1]
    vec_curr_distances_speed = gl_distances_speed[frame_count]
    distances_acceleration = vec_curr_distances_speed - vec_prev_distances_speed
    gl_distances_acceleration.append(distances_acceleration)


def generate_heights_acceleration_matrix():
    if frame_count < 2:
        gl_heights_acceleration.append(None)
        return

    vec_prev_heights_speed = gl_heights_speed[frame_count - 1]
    vec_curr_heights_speed = gl_heights_speed[frame_count]
    heights_acceleration = vec_curr_heights_speed - vec_prev_heights_speed
    gl_heights_acceleration.append(heights_acceleration)


def fill_matrix(coordinates):
    generate_distances_matrix(coordinates)
    generate_heights_matrix(coordinates)
    generate_distances_speed_matrix()
    generate_heights_speed_matrix()
    generate_distances_acceleration_matrix()
    generate_heights_acceleration_matrix()


def get_outlier_value():
    return 0


def fall_detect_pages_algo(svm_value, fall, cu_sum):
    if svm_value >= 0.0:
        if cu_sum >= upper_bound:
            cu_sum = upper_bound
        else:
            cu_sum += svm_value

        if cu_sum == upper_bound:
            fall = True
            return fall, cu_sum
        else:
            return fall, cu_sum
    else:
        if cu_sum + svm_value <= lower_bound:
            cu_sum = lower_bound
        else:
            cu_sum += svm_value

        if cu_sum == lower_bound:
            fall = False
            return fall, cu_sum
        else:
            return fall, cu_sum


def capture_picture():
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


logging.getLogger(comtypes.logger.name).setLevel(logging.CRITICAL)
logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='JSON config')
    parser.add_argument('-v', default=1, help='Kinect data visualisation')
    args = parser.parse_args()

    logger.info(f'Load config from {args.config}')

    with open(args.config, 'r') as stream:
        config = json.load(stream)
        EXCLUDE_SKELETON_DISTANCES = config["EXCLUDE_SKELETON_DISTANCES"]
        SKELETON_COLORS = config["SKELETON_COLORS"]
        SKELETON_USABLE_JOINTS_NUMBER = config["SKELETON_USABLE_JOINTS_NUMBER"]
        SKELETON_LINES = config["SKELETON_LINES"]
        SKELETON_POINTS_COUNT = config["SKELETON_POINTS_COUNT"]
        SKELETON_HEIGHTS_CORRECTION = config["SKELETON_HEIGHTS_CORRECTION"]
    config = args.config

    queue_video_recording = Queue()
    process_video_recording = Process(target=send_preprocessing, args=(queue_video_recording, config))
    process_video_recording.start()

    # queue_video_recording.put(Call.LOGIN)

    # logger.info('Kinect color image saving...')
    # capture_picture()
    # logger.info('Kinect color image saved.')

    # queue_print.put(Call.TEST)
    # queue_print.put(Call.EXIT)
    # while queue_print.qsize() != 0:
    #     pass
    # process_print.join()
    # print("End.")
    # exit(0)
    visual = args.v
    logger.info(f'Starting service...')
    logger.info(f'Connecting to Microsoft Kinect v2')

    with KinectEmulator(os.environ.get('emulator_base_path')) as kinect:
        # KINECT_FREQUENCY = 0.
        #
        # with PyKinectRuntime.PyKinectRuntime(
        #         PyKinectV2.FrameSourceTypes_Color
        #         | PyKinectV2.FrameSourceTypes_Body
        #         | PyKinectV2.FrameSourceTypes_Depth) as kinect:

        kinect_frame = KinectFrame(kinect.max_body_count)
        kinect_frame.set_path(f'recs/{os.environ.get("test_person_name")}/')
        kinect_frame.create_directory()
        bodies = None

        if visual == '1':
            queue_video_streaming = Queue()
            process_video_streaming = Process(target=paint, args=(queue_video_streaming, config))
            process_video_streaming.start()

        skeleton_frame = None

        frame_count = 0
        time.sleep(2)

        queue_video_recording.put((Call.VIDEO_OPEN_STREAMING, [
            kinect_frame.get_path(),
            (kinect.color_frame_desc.Width, kinect.color_frame_desc.Height)
        ]))

        id_delay = 5  # in sec
        monitoring_t = 10

        queue_person_identification = Queue()
        process_person_identification = Process(target=identify, args=(queue_person_identification, config))
        process_person_identification.start()
        queue_person_identification.put(MState.CHECK_STATUS)
        queue_person_identification.put((
            MState.INIT_MONITORING,
            kinect.max_body_count
        ))
        process_id_start = time.time()
        process_id_end = -1
        people_indexes = ['NoID'] * kinect.max_body_count
        monitoring_process = True
        waiting_person_id = True
        while not terminate:
            time.sleep(KINECT_FREQUENCY)
            if True:

                if not monitoring_process:
                    if (time.time() - process_id_end) > id_delay:
                        print('Once again')
                        monitoring_process = True
                        waiting_person_id = True
                        process_person_identification = Process(target=identify,
                                                                args=(queue_person_identification, config))
                        process_person_identification.start()
                        queue_person_identification.put(MState.CHECK_STATUS)
                        queue_person_identification.put((
                            MState.INIT_MONITORING,
                            kinect.max_body_count
                        ))
                        process_id_start = time.time()

                if kinect.has_new_color_frame():
                    kinect_frame.set_color_width(kinect.color_frame_desc.Width)
                    kinect_frame.set_color_height(kinect.color_frame_desc.Height)
                    kinect_frame.color_frame = kinect.get_last_color_frame()

                if kinect.has_new_body_frame():
                    bodies = kinect.get_last_body_frame()

                if bodies is not None:
                    next_frame = False
                    have_bodies = False
                    for i in range(0, kinect.max_body_count):
                        body = bodies.bodies[i]

                        if not body.is_tracked:
                            continue

                        have_bodies = True
                        next_frame = True
                        joints = body.joints  # Joints in a skeleton space
                        joints_color_space = kinect.body_joints_to_color_space(joints)
                        player_coordinates = []
                        player_screen_coordinates = []

                        SKELETON_USABLE_JOINTS_NUMBER = dict(
                            sorted(
                                SKELETON_USABLE_JOINTS_NUMBER.items(),
                                key=lambda x: x[1]
                            )
                        )
                        for j in SKELETON_USABLE_JOINTS_NUMBER.values():
                            player_coordinates.append(
                                types.Joint(joints[j].Position.x * 1000, joints[j].Position.y * 1000,
                                            joints[j].Position.z * 1000, joints[j].TrackingState))

                            if math.isinf(joints_color_space[j].x) and math.isinf(joints_color_space[j].y):
                                player_screen_coordinates.append(types.ScreenJoint(0, 0))
                            else:
                                player_screen_coordinates.append(types.ScreenJoint(int(joints_color_space[j].x),
                                                                                   int(joints_color_space[j].y)))

                        kinect_frame.skeleton_frames[i] = SkeletonFrame(player_coordinates, player_screen_coordinates)
                        if kinect_frame.skeleton_frames[i] is not None:
                            kinect_frame.skeleton_frames[i].set_id(people_indexes[i])

                    if visual == '1':
                        queue_video_streaming.put({i: kinect_frame})
                    if len(kinect_frame.skeleton_frames) > 0 and monitoring_process:
                        # queue_video_recording.put((Call.VIDEO_STREAMING, [
                        #     kinect_frame.skeleton_frames,
                        #     kinect_frame.color_frame,
                        # ]))
                        queue_person_identification.put((MState.APPEND_MONITORING, kinect_frame.skeleton_frames))
                    if round(time.time() - process_id_start) > monitoring_t and monitoring_process:
                        queue_person_identification.put(MState.GET_PERSON_ID)
                        monitoring_process = False
                        process_id_end = time.time()
                        people_indexes = ["Reloading..."] * kinect.max_body_count

                    if (round(time.time() - process_id_start) > monitoring_t + 1
                            and not monitoring_process
                            and waiting_person_id):
                        waiting_person_id = False
                        process_person_identification.join()
                        ids = queue_person_identification.get()
                        for j, person_id in enumerate(ids):
                            people_indexes[j] = person_id
                        print(people_indexes)
                        process_person_identification.terminate()

                    if process_video_streaming.exitcode is not None:

                        while not queue_video_streaming.qsize() == 0:
                            queue_video_streaming.get()
                        queue_video_recording.put(Call.VIDEO_CLOSE_STREAMING)
                        queue_person_identification.put(MState.END_MONITORING)
                        process_video_recording.join()
                        process_person_identification.join()
                        process_video_streaming.join()
                        terminate = True
                    elif i == 5 and not have_bodies:
                        queue_video_streaming.put({i: kinect_frame})

                    if next_frame:
                        frame_count += 1

    # Close Kinect sensor, close the window and quit.
    logger.info(f'Microsoft Kinect v2 was disconnected')
    logger.info(f'Stopping service...')
    logger.info(f'Service stopped.')
