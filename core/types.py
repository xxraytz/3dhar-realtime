import math
import time
from typing import NamedTuple
from svm.svc_addon import Svc
from svm.one_class_addon import OneClass
from multiprocessing import Queue, Process
from pykinect2_emu.pykinect2_emu import KinectEmulator


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

    def tolist(self):
        return [
            self.x, self.y, self.z, float(self.state)
        ]


class SkeletonFrame:

    def __init__(self, coords, screen_coords) -> object:
        """constructor"""
        self.player_coordinates = coords
        self.player_screen_coordinates = screen_coords
        self.player_ID = 'NoID'

    def set_id(self, id):
        self.player_ID = id

    def is_zero_state(self):
        """Check defective skeleton"""
        if len(self.player_coordinates) == 0:
            return None
        for joint in self.player_coordinates:
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

    def classify(self, svm_vector: list):
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

    def __init__(self, max_body_count, color_frame=None):
        self.color_frame = color_frame
        self.depth_frame = []
        self.skeleton_frames = [None] * max_body_count
        self.__depth_frame_height = 0
        self.__depth_frame_width = 0
        self.__color_frame_height = 0
        self.__color_frame_width = 0

        self.__path = ""

    def reset(self, max_body_count):
        self.skeleton_frames = [None] * max_body_count

    def get_color_width(self):
        return self.__color_frame_width

    def get_color_height(self):
        return self.__color_frame_height

    def get_depth_width(self):
        return self.__depth_frame_width

    def get_depth_height(self):
        return self.__depth_frame_height

    def set_color_frame(self, color_frame):
        self.__color_frame_width, self.__color_frame_height = color_frame.shape[:2][::-1]
        self.color_frame = color_frame

    def set_color_width(self, value):
        self.__color_frame_width = value

    def set_color_height(self, value):
        self.__color_frame_height = value

    def set_depth_width(self, value):
        self.__depth_frame_width = value

    def set_depth_height(self, value):
        self.__depth_frame_height = value

    # def save_fall_statements(self):
    #     filename = self.__path + "falls.txt"
    #     with open(filename, "a+", encoding="utf-8") as f:
    #         f.write("Fall_start " + str(self.start_fall) + ", Fall_end: " + str(self.end_fall) + "\n")
    #         self.start_fall = -1
    #         self.end_fall = -1


class KinectFrameReader:
    def __init__(self, kinect : KinectEmulator):
        self.kinect = kinect
        self.bodies = None

    def get_max_body_count(self):
        return self.kinect.max_body_count

    def get_color_size(self):
        return self.kinect.color_frame_desc.Width, self.kinect.color_frame_desc.Height

    def has_color_frame(self):
        return self.kinect.has_new_color_frame()

    def has_body(self):
        return self.kinect.has_new_body_frame()

    def get_body_frame(self):
        return self.kinect.get_last_body_frame()

    def get_color_frame(self):
        if self.has_color_frame():
            return self.kinect.get_last_color_frame()
        else:
            return None

    def set_bodies(self):
        if self.has_body():
            self.bodies = self.get_body_frame()
            return True

    def get_bodies(self):
        return self.bodies.bodies

    def get_body(self, index):
        return self.bodies.bodies[index]

    def is_body_tracked(self, index):
        return self.bodies.bodies[index].is_tracked

    def get_joints(self, index):

        joints = []
        for joint in self.bodies.bodies[index].joints:
            joints.append([joint.Position.x * 1e3,
                           joint.Position.y * 1e3,
                           joint.Position.z * 1e3,
                           joint.TrackingState])

        return joints

    def get_color_space_joints(self, index):
        joints = []
        for joint in self.kinect.body_joints_to_color_space(self.get_body(index).joints):
            if math.isinf(joint.x) and math.isinf(joint.y):
                joints.append([0,0])
            else:
                joints.append([int(joint.x), int(joint.y)])
        return joints


class PersonIdentification:
    def __init__(self, id_delay, t_monitoring):
        self.__delay = id_delay
        self.__t = t_monitoring
        self.__start_time = 0.
        self.__end_time = -1.
        self.__monitoring_process = True
        self.__waiting_person_id = True

        self.__queue_person_identification = Queue()


    def is_monitoring(self):
        return self.__monitoring_process

    def is_waiting_for_identification(self):
        return self.__waiting_person_id

    def stop_waiting_for_identification(self):
        self.__waiting_person_id = False

    def stop_monitoring(self):
        self.__monitoring_process = False
        self.__end_time = time.time()

    def start_id(self):
        self.__start_time = time.time()
        self.__monitoring_process = True
        self.__waiting_person_id = True

    def get_delay(self):
        return self.__delay

    def get_monitoring_period(self):
        return self.__t

    def is_delay_has_passed(self):
        return (time.time() - self.__end_time) > self.__delay

    def is_monitoring_time_passed(self):
        return round(time.time() - self.__start_time) > self.__t and self.is_monitoring()

    def is_identification_delay_passed(self):
        post_monitoring_delay = 4
        return (round(time.time() - self.__start_time) > self.__t + post_monitoring_delay
                and not self.is_monitoring()
                and self.is_waiting_for_identification())

