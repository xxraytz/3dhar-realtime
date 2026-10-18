import numpy as np
from tools.person_height_estimation import PersonHeight


class SkeletonCorrector:
    PERSON_AVERAGE_COUNT = 10

    def __init__(self, mode='3D', skeleton_confidence=None):
        self.mode = mode
        self.conf = skeleton_confidence
        self.transforming_video = None

    def transform(self, skeletal_video):
        self.transforming_video = skeletal_video
        # print(self.estimate_height(self.conf))
        self.height_correction(self.estimate_height(self.conf))
        self.align_on_the_floor()

        if self.mode == '2D':
            self.skeleton_overlay_2d()
        else:
            self.skeleton_overlay()

        return self.transforming_video

    def estimate_height(self, conf=None):
        person_height = PersonHeight('id_01', self.transforming_video)
        return np.mean(person_height.get_heights(self.PERSON_AVERAGE_COUNT, conf))

    GRAY_MAX = 0.86

    def height_correction(self, height):
        for skeleton in self.transforming_video:
            skeleton /= height

    def align_on_the_floor(self):
        for skeleton in self.transforming_video:
            d_y = np.inf
            for joint in skeleton:
                if d_y > joint[1]:
                    d_y = joint[1]
            for joint in skeleton:
                joint[1] -= d_y

    def skeleton_overlay_2d(self):
        for skeleton in self.transforming_video:
            for i, joint in enumerate(skeleton):
                if i != 0:
                    joint[0] -= skeleton[0][0]
                    # joint[1] -= skeleton[0][1]
            skeleton[0][0] = 0
            # skeleton[0][1] = 1

    def skeleton_overlay(self):
        for skeleton in self.transforming_video:
            for i, joint in enumerate(skeleton):
                if i != 0:
                    joint[0] -= skeleton[0][0]
                    # joint[1] -= skeleton[0][1]
                    joint[2] -= skeleton[0][2]
            skeleton[0][0] = 0
            # skeleton[0][1] = 0
            skeleton[0][2] = 0


