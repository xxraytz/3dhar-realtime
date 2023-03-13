import collections
import datetime
import enum
import json
import os

import backoff  # https://pypi.org/project/backoff/
import requests

import logging
from enum import Enum

import cv2 as cv2
import numpy as np
from core.types import KinectFrame
from http import HTTPStatus  # https://docs.python.org/3/library/http.html

# SKELETON_POINTS_COUNT = 17

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s', datefmt='%H:%M:%S')


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


def refresh_tokens(config, access=None, refresh=None):
    with open(config, 'r+') as stream:
        conf = json.load(stream)
        if access:
            conf["TOKEN_ACCESS"] = access
        if refresh:
            conf["TOKEN_REFRESH"] = refresh
        stream.seek(0)
        json.dump(conf, stream, sort_keys=True, indent=4, separators=(',', ': '))


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_time=60)
def send_data_to_server(video_path, skeletal_video_path, config):
    with open(config, 'r') as stream:
        conf = json.load(stream)
        token_access = conf["TOKEN_ACCESS"]
        token_refresh = conf["TOKEN_REFRESH"]
        POST_VERIFY_URL = f'{conf["API_URL"]}token-verify/'
        POST_REFRESH_URL = f'{conf["API_URL"]}refresh-token/'
        POST_URL = f'{conf["API_URL"]}incident/'

    with requests.Session() as session:
        post = session.post(POST_VERIFY_URL, json={"token": token_access})
        if post.status_code != HTTPStatus.OK:
            post = session.post(POST_REFRESH_URL, json={"refresh": token_refresh})
            token_access = json.loads(post.text)
            refresh_tokens(config, access=token_access)

        headers = {'Authorization': 'Bearer ' + token_access}
        post = session.post(
            POST_URL, headers=headers,
            data={
                'start_time': str(datetime.datetime.now()),
                'end_time': str(datetime.datetime.now())
            }, files={
                'real_video': (video_path, open(video_path, 'rb')),
                'skeletal_video': (skeletal_video_path, open(skeletal_video_path, 'rb')),
            })
        response = f'Request status code: {post.status_code!s}'
        if post.status_code != HTTPStatus.CREATED:
            logger.info(response)
            raise requests.exceptions.RequestException
        return response


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_time=60)
def auth(config):
    with open(config, 'r+') as stream:
        conf = json.load(stream)
        POST_LOGIN_URL = f'{conf["API_URL"]}login/'
        with requests.Session() as session:
            post = session.post(POST_LOGIN_URL,
                                json={"email": os.environ.get("email"), "password": os.environ.get("password")})
            response = f'Request status code: {post.status_code!s}'
            if post.status_code != HTTPStatus.OK:
                logger.info(response)
                raise requests.exceptions.RequestException
            token = json.loads(post.text)
        refresh_tokens(config=config, access=f'{token["access"]}', refresh=f'{token["refresh"]}')
        return response


def put_skeleton_to_image(saved_color_image, skeleton, config):
    skeleton_rgb = (0, 0, 255)
    with open(config, 'r') as stream:
        config = json.load(stream)
        skeleton_lines = config["SKELETON_LINES"]

    for line in skeleton_lines:
        cv2.line(saved_color_image,
                 skeleton[line[0]], skeleton[line[1]], skeleton_rgb, 5)

    return saved_color_image


def send_preprocessing(q_print, config):
    video_resolution = (0, 0)
    skeleton_sequence = collections.deque(maxlen=300)
    real_video_sequence = collections.deque(maxlen=300)
    video_stream_writing = None
    video_stream_path = f'person_stream.mp4'
    while True:
        data = q_print.get()
        if data is None:
            break
        if isinstance(data, tuple):
            mode = data[0]
        else:
            mode = data

        # server communication tests
        if mode.value == Call.LOGIN.value:
            auth(config)
            print("ev is ok")
            # break
        elif mode.value == Call.TEST.value:
            send_data_to_server("tests/video-res/video.mp4", "tests/video-res/video.mp4", config)
            logger.info("Sent.")
            # break
        elif mode.value == Call.EXIT.value:
            # data = q_print.get()
            print('Exiting')
            break
        elif mode.value is Call.VIDEO_OPEN_STREAMING.value:
            logger.debug("Open streaming...")
            print("Open streaming...")
            video_resolution,  = data[1]
            print(video_resolution)
            fourcc = cv2.VideoWriter_fourcc(*'h264')
            video_stream_writing = cv2.VideoWriter(
                video_stream_path,
                fourcc,
                24,
                video_resolution,
            )
        elif mode.value is Call.VIDEO_STREAMING.value:
            skeleton, real_video_frame = data[1]
            image = np.reshape(real_video_frame, (*reversed(video_resolution), 4))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            video_stream_writing.write(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB))
        elif mode.value is Call.VIDEO_CLOSE_STREAMING.value:
            video_stream_writing.release()
            break
        elif mode.value == Call.VIDEO_APPEND.value:
            skeleton, real_video, video_resolution = data[1]
            skeleton_sequence.append(skeleton)
            real_video_sequence.append(real_video)
    print("Out of cycle")


# Drop skeleton points to file
def drop_skeleton_to_file(file, frame_index, player_joints, player_joint_cs):
    filename = file + "Body//FileSkeleton.txt"
    with open(filename, "a+", encoding="utf-8") as f:
        f.write("Frame: {0:d}\n".format(frame_index))
        logger.info(f'{len(player_joints)}')
        for ind in range(len(player_joints)):
            f.write(f'{player_joints[ind].x:0.3f} '
                    f'{player_joints[ind].y:1.3f}'
                    f'{player_joints[ind].z:2.3f}'
                    f'{player_joints[ind].state:3d}'
                    f'{player_joint_cs[ind].x:4.3f}'
                    f'{player_joint_cs[ind].y:5.3f}')

    logger.info(f'Body was successfully written to file ' + filename)
