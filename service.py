import argparse
import json
import logging
import math
import os
from datetime import datetime

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
from core.types import KinectFrameReader
from core.types import PersonIdentification

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
    max_height = coordinates[SKELETON_USABLE_JOINTS_NUMBER["Head"]].y + SKELETON_HEIGHTS_CORRECTION

    for frame in skeleton_frames:
        h = frame.get_coordinates()[SKELETON_USABLE_JOINTS_NUMBER["Head"]].y + SKELETON_HEIGHTS_CORRECTION
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


def get_config(args):
    global EXCLUDE_SKELETON_DISTANCES, SKELETON_COLORS, SKELETON_USABLE_JOINTS_NUMBER, SKELETON_LINES, SKELETON_POINTS_COUNT, SKELETON_HEIGHTS_CORRECTION
    with open(args.config, 'r') as stream:
        config = json.load(stream)
        EXCLUDE_SKELETON_DISTANCES = config["EXCLUDE_SKELETON_DISTANCES"]
        SKELETON_COLORS = config["SKELETON_COLORS"]
        SKELETON_USABLE_JOINTS_NUMBER = config["SKELETON_USABLE_JOINTS_NUMBER"]
        SKELETON_USABLE_JOINTS_NUMBER = dict(
            sorted(
                SKELETON_USABLE_JOINTS_NUMBER.items(),
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


if __name__ == '__main__':
    args = parse_arguments()

    logger.info(f'Load config from {args.config}')

    config = get_config(args)

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

    person_identification = PersonIdentification(id_delay=5, t_monitoring=10)

    with KinectEmulator(os.environ.get('emulator_base_path')) as kinect:
        #
        # with PyKinectRuntime.PyKinectRuntime(
        #         PyKinectV2.FrameSourceTypes_Color
        #         | PyKinectV2.FrameSourceTypes_Body
        #         | PyKinectV2.FrameSourceTypes_Depth) as kinect:
        kinect_device = KinectFrameReader(kinect)
        store_directory = create_store_directory(f'recs/{os.environ.get("test_person_name")}/')
        bodies = None

        if visual == '1':
            queue_video_streaming = Queue()
            process_video_streaming = Process(target=paint, args=(queue_video_streaming, config))
            process_video_streaming.start()

        skeleton_frame = None

        frame_count = 0
        time.sleep(2)

        queue_video_recording.put((Call.VIDEO_OPEN_STREAMING, [
            store_directory,
            kinect_device.get_color_size()
        ]))

        queue_person_identification = Queue()
        process_person_identification = Process(target=identify, args=(queue_person_identification, config))
        process_person_identification.start()

        queue_person_identification.put(MState.CHECK_STATUS)
        queue_person_identification.put((MState.INIT_MONITORING, kinect.max_body_count))
        queue_person_identification.put((MState.GET_PEOPLE,))

        person_identification.start_id()

        people_indexes = ['NoID'] * kinect.max_body_count

        while not terminate:
            time.sleep(KINECT_FREQUENCY)
            if True:

                if not person_identification.is_monitoring():
                    if person_identification.is_delay_has_passed():
                        process_person_identification = Process(target=identify,
                                                                args=(queue_person_identification, config))
                        process_person_identification.start()

                        queue_person_identification.put(MState.CHECK_STATUS)
                        queue_person_identification.put((MState.INIT_MONITORING, kinect_device.get_max_body_count()))
                        person_identification.start_id()

                kinect_frame = KinectFrame(kinect_device.get_max_body_count())
                kinect_frame.set_color_frame(kinect_device.get_color_frame())

                if kinect_device.set_bodies():
                    have_bodies = False
                    for body_index in range(0, kinect_device.get_max_body_count()):
                        if kinect_device.is_body_tracked(body_index):
                            continue

                        have_bodies = True

                        joints_skeleton_space = kinect_device.get_joints(body_index)  # Joints in a skeleton space
                        joints_color_space = kinect_device.get_color_space_joints(body_index)

                        player_coordinates = []
                        player_screen_coordinates = []

                        for j in SKELETON_USABLE_JOINTS_NUMBER.values():
                            player_coordinates.append(types.Joint(*joints_skeleton_space[j]))

                            player_screen_coordinates.append(types.ScreenJoint(*joints_color_space[j]))

                        kinect_frame.skeleton_frames[body_index] = SkeletonFrame(player_coordinates,
                                                                                 player_screen_coordinates)
                        kinect_frame.skeleton_frames[body_index].set_id(people_indexes[body_index])

                    if visual == '1':
                        queue_video_streaming.put({body_index: kinect_frame})

                    if len(kinect_frame.skeleton_frames) > 0 and person_identification.is_monitoring():
                        # queue_video_recording.put((Call.VIDEO_STREAMING, [
                        #     kinect_frame.skeleton_frames,
                        #     kinect_frame.color_frame,
                        # ]))
                        queue_person_identification.put((MState.APPEND_MONITORING, kinect_frame.skeleton_frames))

                    if person_identification.is_monitoring_time_passed():
                        queue_person_identification.put(MState.GET_PERSON_ID)
                        person_identification.stop_monitoring()
                        people_indexes = ["Reloading..."] * kinect.max_body_count

                    if person_identification.is_identification_delay_passed():
                        person_identification.stop_waiting_for_identification()
                        process_person_identification.join()
                        ids = queue_person_identification.get()
                        for j, person_id in enumerate(ids):
                            people_indexes[j] = person_id
                        logger.info(people_indexes)
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
                    elif body_index == 5 and not have_bodies:
                        queue_video_streaming.put({body_index: kinect_frame})

                    if have_bodies:
                        frame_count += 1

    # Close Kinect sensor, close the window and quit.
    logger.info(f'Microsoft Kinect v2 was disconnected')
    logger.info(f'Stopping service...')
    logger.info(f'Service stopped.')
