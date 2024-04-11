import numpy as np
from scipy import io


class SkeletonReader:

    def __init__(self, video_path, m, dim):
        self.video_path = video_path
        self.database_name = m
        self.__dim = dim

    def read_skeletons(self, basic_assembly_reading=False):
        if basic_assembly_reading:
            return self.__read_basic_assembly()
        if self.database_name == 'SYSU':
            return self.__read_skeletal_sequence_sysu_db()
        elif self.database_name == 'PKU':
            return self.__read_skeletal_sequence_pku_db()
        elif self.database_name == 'TST':
            return self.__read_skeletal_sequence_default()
        elif self.database_name == 'TST_2D':
            return self.__read_skeletal_sequence_default_2d()

    def __read_skeletal_sequence_sysu_db(self):
        skeleton_mat = io.loadmat(self.video_path)
        skeleton_mat = skeleton_mat['SW'] * 1000
        video = []
        for i in range(skeleton_mat.shape[-1]):
            skeleton = np.delete(skeleton_mat[:, :, i], [18, 14, 10, 6], axis=0)
            skeleton = np.vstack((skeleton,
                                  np.mean([skeleton[2], skeleton[3]], axis=0)))
            temp_point = skeleton[2, :].copy()
            skeleton[2, :] = skeleton[16, :]
            skeleton[16, :] = temp_point
            video.append(skeleton)
        return video if video else -1

    def __read_skeletal_sequence_default(self):
        with open(self.video_path, 'r+') as file:
            video = []
            for line in file:
                if 'frame' in line.lower():
                    skeleton = np.zeros(shape=(0, 4))
                    for i in range(17):
                        skeleton = np.vstack([
                            skeleton,
                            np.asarray([
                                [float(el) for el in file.readline().split(', ')[:4]]
                            ])
                        ])
                    video.append(skeleton.copy())
        if self.__dim == '2D':
            return [el[:, :2] for el in video]
        else:
            return [el[:, :-1] for el in video]

    def __read_skeletal_sequence_default_2d(self):
        with open(self.video_path, 'r+') as file:
            video = []
            for line in file:
                if 'frame' in line.lower():
                    skeleton = np.zeros(shape=(0, 3))
                    for i in range(17):
                        skeleton = np.vstack([
                            skeleton,
                            np.asarray([
                                [float(el) for el in file.readline().split(', ')[:4]]
                            ])
                        ])
                    video.append(skeleton.copy())
        return video if video else -1

    def __read_skeletal_sequence_pku_db(self):
        video = np.loadtxt(self.video_path, delimiter=' ')
        skeleton_kinect = {}
        key_points = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10,
                      12, 13, 14, 16, 17, 18, 20, ]
        for i, line in enumerate(video):
            skeleton_kinect[f'frame {i}'] = []
            if np.sum(np.abs(line[:75])):
                skeleton_kinect[f'frame {i}'] = line[:75].reshape(25, 3)[np.ix_(key_points)]
        if self.__dim == '2D':
            return [el[:, :2] for el in list(skeleton_kinect.values())]
        else:
            return list(skeleton_kinect.values())

    def __read_basic_assembly(self):
        with open(self.video_path, 'r+') as file:
            video = []
            for line in file:
                if 'frame' in line.lower():
                    skeleton = np.zeros(shape=(0, 4))
                    for i in range(17):
                        skeleton = np.vstack([
                            skeleton,
                            np.asarray([
                                [float(el) for el in file.readline().split(', ')[:4]]
                            ])
                        ])
                    video.append(skeleton.copy())
        return video