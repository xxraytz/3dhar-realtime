import logging
import os
from collections import deque
from enum import Enum
from core.measurements import knn, calc_measure_svc, debug_calc_measure, debug_calc_measure_svc
from core.measurements import calc_measure, get_train_params
import numpy as np
from core.skeleton_shifting import height_correction, align_on_the_floor, skeleton_overlay, basic_assembly_metric
from core.skeleton_zero_view import skeleton_correction
import pickle

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


class HAR:

    def __init__(self, body_number):

        self.skeletons = {}
        self.database = None
        self.train_x = None
        self.train_y = None
        self.classes = None
        self.__body_count = body_number
        self.__people_indexes = ['NoID'] * self.__body_count
        self.svm_classifier = None
        self.one_class_classifier = None

    def database_connect(self):
        self.database = get_database(os.environ.get('database_path'))
        self.database = [el[:, :-1] for el in self.database]
        self.database = height_correction(self.database, DATABASE_HEIGHT)
        self.database = align_on_the_floor(self.database)
        self.database = skeleton_overlay(self.database)
        with open('C:/Users/Egor/workspace/projects/python/person-skeleton-id/data/svm-new.pickle', 'rb') as handle:
            self.svm_classifier = pickle.load(handle)
        with open('C:/Users/Egor/workspace/projects/python/person-skeleton-id/data/one-class-elliptic-c=0.13-sf=0.9.pickle', 'rb') as handle:
            self.one_class_classifier = pickle.load(handle)
        self.reset_train_set()

    def reset_train_set(self):
        self.train_x, self.train_y = get_profiles()
        self.train_x = self.train_x[:, [4, 10, 11, 15, 28]]
        self.classes = np.unique(self.train_y)

    def select_people_in_train_set(self, people_list):
        self.reset_train_set()
        self.train_x = self.train_x[np.where(self.train_y == np.array(people_list)[:, None])[1]]
        self.train_y = self.train_y[np.where(self.train_y == np.array(people_list)[:, None])[1]]
        self.classes = np.unique(self.train_y)

    def start_monitoring(self):
        for i in range(0, self.__body_count):
            self.skeletons[f'skeleton_{i:03d}'] = deque(maxlen=500)

    def append_frame(self, frame):
        for i, sk_ind in enumerate(frame):
            if sk_ind is not None:
                self.skeletons[f'skeleton_{i:03d}'].append(sk_ind)

    def get_person_id(self, mon_people=None):
        pass


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
