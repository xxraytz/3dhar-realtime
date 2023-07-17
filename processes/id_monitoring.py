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
    APPEND_MONITORING = "append"
    END_MONITORING = "end"
    GET_PERSON_ID = "get_id"
    CHECK_STATUS = "check_status"
    GET_PEOPLE = "get_all_people"
    INIT_MONITORING = "initialization"


class PersonIdentification:

    def __init__(self, body_number):
        self.skeletons = {}
        self.database = None
        self.train_x = None
        self.train_y = None
        self.__body_number = body_number
        self.__people_indexes = ['NoID'] * self.__body_number

    def database_connect(self):
        self.database = get_database(os.environ.get('database_path'))
        self.database = [el[:, :-1] for el in self.database]
        self.database = height_correction(self.database, DATABASE_HEIGHT)
        self.database = align_on_the_floor(self.database)
        self.database = skeleton_overlay(self.database)

        self.train_x, self.train_y, _ = get_profiles()
        self.train_x = self.train_x[:, [4, 10, 11, 15, 28]]

    def start_monitoring(self):
        for i in range(0, self.__body_number):
            self.skeletons[f'skeleton_{i:03d}'] = deque(maxlen=500)

    def append_frame(self, frame):
        for i, sk_ind in enumerate(frame):
            if sk_ind is not None:
                self.skeletons[f'skeleton_{i:03d}'].append(sk_ind)

    def get_person_id(self):
        ids = ['NoID'] * self.__body_number
        for i, sk_ind in enumerate(self.skeletons):
            if len(self.skeletons[sk_ind]) > 10:
                video = list(map(lambda x:
                                 np.asarray(list(map(lambda y: y.tolist(), x.player_coordinates))),
                                 self.skeletons[sk_ind]))
                video = [el[:, :-1] for el in video]
                video = skeleton_correction(video)
                video = height_correction(video, VIDEO_HEIGHT)
                video = align_on_the_floor(video)
                video = skeleton_overlay(video)
                dm = basic_assembly_metric(self.database, video)
                test_x = div_profiles(dm)
                try:
                    test_x = test_x[:, [4, 10, 11, 15, 28]]
                except IndexError:
                    print(test_x)
                # ids[i] = get_person_id(profile, database_profiles)
                classes = np.unique(self.train_y)
                ids[i] = calc_measure(test_x, *get_train_params(self.train_x, self.train_y, classes),
                                      classes, measure='euclidean')
        return ids

    def get_people(self):
        return np.unique(self.train_y)


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
