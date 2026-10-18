import numpy as np
from scipy.spatial import distance


def height_correction(sample_video, height):
    for skeleton in sample_video:
        for joint in skeleton:
            joint[0] /= height
            joint[1] /= height
            joint[2] /= height
    return sample_video


def align_on_the_floor(sample_video):
    for skeleton in sample_video:
        d_y = np.inf
        for joint in skeleton:
            if d_y > joint[1]:
                d_y = joint[1]
        for joint in skeleton:
            joint[1] -= d_y
    return sample_video


def skeleton_overlay(sample_video):
    for skeleton in sample_video:
        for i, joint in enumerate(skeleton):
            if i != 0:
                joint[0] -= skeleton[0][0]
                joint[2] -= skeleton[0][2]
        skeleton[0][0] = 0
        skeleton[0][2] = 0
    return sample_video


def basic_assembly_metric(basic_assembly, video_sequence):
    # def skeletal_distance(s1, s2):
    #     res = 0
    #     for x, y in zip(s1, s2):
    #         res += distance.euclidean(x, y)
    #     return res / 17.
    #
    # distance_matrix = np.zeros(shape=(len(basic_assembly), 0))
    # for i, skeleton_frame in enumerate(video_sequence):
    #     d = np.asarray(list(map(lambda ba: skeletal_distance(ba, skeleton_frame), basic_assembly)))
    #     d = np.expand_dims(d.T, axis=1)
    #     distance_matrix = np.hstack([distance_matrix, d])
    distance_matrix = np.mean(np.linalg.norm(np.asarray(basic_assembly)[:, None] - np.asarray(video_sequence)[None, :],
                                             axis=-1), axis=-1)
    return distance_matrix
