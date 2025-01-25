import logging
from collections import deque
import numpy as np
from tools.skeleton_reader import SkeletonReader
from tools.skeleton_corrector import SkeletonCorrector
from os import environ
from tools.nn_model import Recognizer

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HAR:

    def __init__(self, body_number, dimension='3D', color_style='GRAY', ):
        self.__ba_path = environ.get('ba_path')
        self.__order_path = environ.get('order_path')
        self.__dim = dimension
        self.database = None
        self.act_classes = ['standing', 'walking', 'sitting', 'falling', 'grasp', 'ungrasp',
                            'laying', 'post-fall-laying', 'no_action']
        self.fall_classes = ['FALL', 'ADL']
        self.__body_count = body_number
        self.__basic_assembly = None
        self.__nn_model = None
        self.ready_to_reco = False
        self.skeletons = {}

    def database_connect(self):
        logger.info('Database connection')
        self.__basic_assembly = self.__set_basic_assembly()
        self.__set_nn_model()

    def __set_nn_model(self):
        self.fall_recognizer = Recognizer(environ.get('nn_model_path'))
        self.act_recognizer = Recognizer(environ.get('nn_model_fall_check_path'))

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
            self.skeletons[f'skeleton_{i:03d}'] = deque(maxlen=32)

    def append_frame(self, frame):
        for i, sk_ind in enumerate(frame):
            if sk_ind is not None:
                self.skeletons[f'skeleton_{i:03d}'].append(sk_ind)

    def get_person_class(self):
        classes = ['no_action'] * 6
        classes_fall_action = ['no_action'] * 6
        for i, sk_ind in enumerate(self.skeletons):
            if len(self.skeletons[sk_ind]) == 32:
                video = list(map(lambda x:
                                 np.asarray(list(map(lambda y: y.tolist(), x.player_coordinates))),
                                 self.skeletons[sk_ind]))
                video = [el[:, :-1] for el in video]
                video = np.asarray(SkeletonCorrector(self.__dim).transform(video))
                dm = self.__dm_calculating(video)
                if dm is None:
                    classes[i] = 'no_action'
                else:
                    result_fall = self.fall_recognizer.predict_actions(dm[None, :])
                    result_act = self.act_recognizer.predict_actions(dm[None, :])
                    classes[i] = self.fall_classes[np.argmax(result_fall)] if result_fall is not None else 'no_action'
                    if (classes[i] == 'FALL' and self.act_classes[np.argmax(result_act)] != 'falling'):
                        classes[i] = 'test'

        return classes

    def __dm_calculating(self, video):
        gray_max = 0.86
        #  Here is a mistake. You should define video size
        if video.shape[0] < 32:
            return None
        distance_matrix = np.mean(np.linalg.norm(self.__basic_assembly[:, None] - video[None, :], axis=-1), axis=-1)
        return distance_matrix / gray_max
