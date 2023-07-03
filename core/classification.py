from svm.svc_addon import Svc
from svm.one_class_addon import OneClass


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



# CONSTANTS
KINECT_FREQUENCY = 0.033
SKELETON_POINTS_COUNT: int
SKELETON_COLORS = []
USABLE_SK_JOINTS = {}
SKELETON_LINES = []
EXCLUDE_SKELETON_DISTANCES = []
SKELETON_HEIGHTS_CORRECTION: int
# host_image = np.array([0, 0])


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
frame_count = 0

def get_max_height(coordinates):
    max_height = coordinates[USABLE_SK_JOINTS["Head"]].y + SKELETON_HEIGHTS_CORRECTION

    for frame in skeleton_frames:
        h = frame.get_coordinates()[USABLE_SK_JOINTS["Head"]].y + SKELETON_HEIGHTS_CORRECTION
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
