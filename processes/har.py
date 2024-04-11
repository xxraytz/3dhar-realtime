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
from sklearn.metrics import pairwise_distances

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATABASE_HEIGHT = 1698
VIDEO_HEIGHT = 1900

class HAR:

    def __init__(self, body_number):
        self.skeletons = {}
        self.classes = None
        self.__body_count = body_number

    def database_connect(self):
        ...

    def start_monitoring(self):
        for i in range(0, self.__body_count):
            self.skeletons[f'skeleton_{i:03d}'] = deque(maxlen=500)

    def append_frame(self, frame):
        for i, sk_ind in enumerate(frame):
            if sk_ind is not None:
                self.skeletons[f'skeleton_{i:03d}'].append(sk_ind)

    def get_person_id(self, mon_people=None):
        ...