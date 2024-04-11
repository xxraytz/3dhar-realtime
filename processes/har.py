import logging
from collections import deque
import numpy as np
from tools.skeleton_reader import SkeletonReader
from tools.skeleton_corrector import SkeletonCorrector
from os import environ
logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HAR:

    def __init__(self, body_number, dimension='3D', color_style='GRAY', ):
        self.__ba_path = environ.get('ba_path')
        self.__order_path = environ.get('order_path')
        self.__dim = dimension
        self.database = None
        self.classes = None
        self.__body_count = body_number
        self.__basic_assembly = None
        self.__nn_model = None
        self.ready_to_reco = False
        self.skeletons = {}

    def database_connect(self):
        logger.info('Database connection')
        self.__basic_assembly = self.__set_basic_assembly()
        self.__nn_model = self.__set_nn_model()

    def __set_nn_model(self):
        return None

    def __set_basic_assembly(self):
        ba = SkeletonReader(self.__ba_path, 'TST', dim='3D').read_skeletons(basic_assembly_reading=True)

        if self.__order_path:
            ba = self.__clean_basic_assembly(ba)

        ba = [el[:, :-1] for el in ba]

        return np.asarray([el[:, :-1] for el in SkeletonCorrector().transform(ba)] if self.__dim == '2D'
                          else SkeletonCorrector().transform(ba))

    def __clean_basic_assembly(self, ba):
        with open(self.__order_path, 'r+') as file:
            ba = np.asarray(ba)[[int(el) for el in file.readline().split(',')]]
        return [row for row in ba]

    def start_monitoring(self):
        for i in range(0, self.__body_count):
            self.skeletons[f'skeleton_{i:03d}'] = deque(maxlen=500)

    def append_frame(self, frame):
        for i, sk_ind in enumerate(frame):
            if sk_ind is not None:
                self.skeletons[f'skeleton_{i:03d}'].append(sk_ind)

    def get_person_id(self, mon_people=None):
        ...