import os
from typing import NamedTuple
from svm.svc_addon import Svc
from svm.one_class_addon import OneClass
from datetime import datetime


class ScreenJoint(NamedTuple):
    """Coordinates for paint skeleton joints on a screen"""
    x: int = 0
    y: int = 0


class Joint(NamedTuple):
    """Skeleton coordinate with statements of confidence"""
    x: float = 0.
    y: float = 0.
    z: float = 0.
    state: int = 0


class SkeletonFrame:
    """Class skeleton"""

    def __init__(self, coords, screen_coords):
        """constructor"""
        self.player_coordinates = coords
        self.player_screen_coordinates = screen_coords
        self.player_ID = 'NoID'

    def is_zero_state(self):
        """Check defective skeleton"""
        if len(self.__player_coordinates) == 0:
            return None
        for joint in self.__player_coordinates:
            if joint.state == 0:
                return False
        return True


class SvmWrapper:

    __path = ""
    __model_main = None
    __model_ourliers = None
    __averages = []
    __rmsd = []

    def __init__(self):
        self.__path = "data/"
        files = {"modelMain": self.__path + "model-main.txt",
                 "modelOutliers": self.__path + "model-outliers.txt",
                 "averages": self.__path + "average.csv",
                 "RMSD": self.__path + "dispersion.csv"}
        self.__averages = self.load_compensation(files.get("averages"))
        self.__rmsd = self.load_compensation(files.get("RMSD"))
        self.c_svc = Svc(C=10000, gamma=0.0001)
        self.c_svc.load_model(files.get("modelMain"))
        self.one_class_svm = OneClass(gamma=0.0001, nu=0.03)
        self.one_class_svm.load_model(files.get("modelOutliers"))

    def get_instance(self):
        pass

    def classify(self, svm_vector:list):
        """

        :param svm_vector:
            dictionary of svm nodes
        :return:
        """
        svm_vector = self.__compensate(svm_vector)
        outlier = self.__is_outlier(svm_vector)
        if outlier:
            return [bool(outlier), 0]

        predict = self.c_svc.predict([self.svm_vector_to_list(svm_vector)])
        return [bool(outlier), -1 if predict == 2 else 1]

    def __compensate(self, svm_vector):
        # Do with simple for cycle.
        # TODO: Check that dict have index ZERO
        for i in range(len(svm_vector)):
            svm_vector[i] = (svm_vector[i] - self.__averages[i]) / self.__rmsd[i]
        return svm_vector

    def __is_outlier(self, vector):
        predict = self.one_class_svm.predict([vector])
        return predict == -1

    @staticmethod
    def svm_vector_to_list(svm_vector: list):
        list_x = [v for v in svm_vector]
        return list_x

    @staticmethod
    def load_compensation(path):
        with open(path, "r") as file:
            data = file.readline().rstrip(",")
            res = [float(number) for number in data.split(",")]
        return res


class KinectFrame:
    """This class make kinect data drawing easier."""

    color_frame = []
    depth_frame = []
    skeleton_frames = []

    def __init__(self):
        __depth_frame_height = 0
        __depth_frame_width = 0
        __color_frame_height = 0
        __color_frame_width = 0

        __path = ""

    def get_color_width(self):
        return self.__color_frame_width

    def get_color_height(self):
        return self.__color_frame_height

    def get_depth_width(self):
        return self.__depth_frame_width

    def get_depth_height(self):
        return self.__depth_frame_height

    def set_color_width(self, value):
        self.__color_frame_width = value

    def set_color_height(self, value):
        self.__color_frame_height = value

    def set_depth_width(self, value):
        self.__depth_frame_width = value

    def set_depth_height(self, value):
        self.__depth_frame_height = value

    def create_directory(self):
        os.mkdir(self.__path)
        os.mkdir(self.__path + "Body/")
        os.mkdir(self.__path + "Color/")
        os.mkdir(self.__path + "Depth/")

    def set_path(self, file):
        print(file)
        print(os.path.isdir(file))
        if not os.path.isdir(file):
            os.mkdir(file)
        print(os.path.isdir(file))
        self.__path = file + f'Scene_{datetime.now().date()}_' \
                             f'{datetime.now().time().hour}-' \
                             f'{datetime.now().time().minute}-'\
                             f'{datetime.now().time().second}' + "/"
        print(self.__path)
        print()

    def get_path(self):
        return self.__path

    def save_fall_statements(self):
        filename = self.__path + "falls.txt"
        with open(filename, "a+", encoding="utf-8") as f:
            f.write("Fall_start " + str(self.start_fall) + ", Fall_end: " + str(self.end_fall) + "\n")
            self.start_fall = -1
            self.end_fall = -1