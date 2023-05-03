import logging
import os
from collections import deque
from enum import Enum
from core.measurements import knn
from core.measurements import calc_measure, get_train_params
from scipy.spatial import distance
import numpy as np
from core.skeleton_shifting import height_correction, align_on_the_floor, skeleton_overlay, basic_assembly_metric
from core.skeleton_zero_view import skeleton_correction

from sklearn.metrics import pairwise_distances

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATABASE_HEIGHT = 1698
VIDEO_HEIGHT = 1900


class MState(Enum):
    START_MONITORING = "start"
    APPEND_MONITORING = "start"
    END_MONITORING = "end"
    GET_PERSON_ID = "get_id"
    CHECK_STATUS = "check_status"
    INIT_MONITORING = "initialization"


def identify(qm, config):
    skeletons = {}
    database = None
    train_x, train_y = None, None
    while True:

        data = qm.get()
        if isinstance(data, tuple):
            state = data[0]
        else:
            state = data

        if state.value is MState.CHECK_STATUS.value:
            logger.info(state.value)
        elif state.value is MState.INIT_MONITORING.value:
            logger.info(state.value)
            database = get_database(os.environ.get('database_path'))
            database = [el[:, :-1] for el in database]
            database = height_correction(database, DATABASE_HEIGHT)
            database = align_on_the_floor(database)
            database = skeleton_overlay(database)

            train_x, train_y, _ = get_profiles()
            train_x = train_x[:, [4, 10, 11, 15, 28]]

            for i in range(0, data[1]):
                skeletons[f'skeleton_{i:03d}'] = deque(maxlen=500)
        elif state.value is MState.END_MONITORING.value:
            break
        elif state.value is MState.APPEND_MONITORING.value:
            skeletons_frame = data[1]
            for i, sk_ind in enumerate(skeletons_frame):
                if sk_ind is not None:
                    skeletons[f'skeleton_{i:03d}'].append(sk_ind)
        elif state.value is MState.GET_PERSON_ID.value:
            logger.info("Getting person ID")
            ids = ['NoID'] * len(skeletons)
            for i, sk_ind in enumerate(skeletons):
                if len(skeletons[sk_ind]) > 10:
                    video = list(map(lambda x:
                                     np.asarray(list(map(lambda y: y.tolist(), x.player_coordinates))),
                                     skeletons[sk_ind]))
                    video = [el[:, :-1] for el in video]
                    video = skeleton_correction(video)
                    video = height_correction(video, VIDEO_HEIGHT)
                    video = align_on_the_floor(video)
                    video = skeleton_overlay(video)
                    dm = basic_assembly_metric(database, video)
                    test_x = div_profiles(dm)
                    test_x = test_x[:, [4, 10, 11, 15, 28]]
                    # ids[i] = get_person_id(profile, database_profiles)
                    classes = np.unique(train_y)
                    ids[i] = calc_measure(test_x, *get_train_params(train_x, train_y, classes),
                                          classes, measure='mahalanobis')
            qm.put(ids)
            break


def get_person_id(cur, stored):
    train_x, train_y, ro_values = stored
    return knn(np.asarray(train_x, dtype=np.float64), np.asarray(train_y),
               np.asarray(cur, dtype=np.float64), ro_values, 3)


def div_profiles(dm):
    t = 100
    N = dm.shape[1] // t
    dm = dm[:, :t * N]
    points = dm.reshape([dm.shape[0], N, t])
    profiles = []
    for i in range(points.shape[1]):
        video = points[:, i, :]
        profiles.append(np.median(video, axis=1))
    return np.asarray(profiles)


def get_profiles():
    with open(os.environ.get("profiles_path"), "r+") as file:
        profiles = []
        profile_names = []
        file.readline()
        for line in file:
            profile_names.append(line.split(",")[:1][0])
            profiles.append(np.asarray(line.split(',')[2:], dtype=np.float32))
    ro_values = {}
    for name in set(profile_names):
        indexes = [index for index, value in enumerate(profile_names) if value == name]
        values = [profiles[i] for i in indexes]
        d = pairwise_distances(values)
        ro_values[name] = float(os.environ.get('alpha')) * np.max(d[np.triu_indices(d.shape[0], 1)])
    return np.asarray(profiles), np.asarray(profile_names), ro_values


def get_database(file):
    with open(file, 'r+') as f:
        video = []
        for line in f:
            if 'frame' in line.lower():
                skeleton = np.zeros(shape=(0, 4))
                for i in range(17):
                    skeleton = np.vstack([
                        skeleton,
                        np.asarray([
                            float(el) for el in f.readline().split(', ')
                        ])
                    ])
                video.append(skeleton.copy())
    return video
