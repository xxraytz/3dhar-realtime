import logging
import os
from collections import deque
from enum import Enum
from scipy.spatial import distance
import numpy as np
from scipy.spatial.distance import cdist
from core.types import Joint

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MState(Enum):
    START_MONITORING = "start"
    APPEND_MONITORING = "start"
    END_MONITORING = "end"
    GET_PERSON_ID = "get_id"
    CHECK_STATUS = "check_status"
    INIT_MONITORING = "initialization"


def identify(qm, config):
    skeletons = {}
    database = None
    profiles = None
    while True:

        data = qm.get()
        if isinstance(data, tuple):
            state = data[0]
        else:
            state = data

        if state.value is MState.CHECK_STATUS.value:
            logger.info(state.value)
        elif state.value is MState.INIT_MONITORING.value:
            logger.info(state.value)
            database = get_database(os.environ.get('database_path'))
            database = height_correction(database)
            database = align_on_the_floor(database)
            database = skeleton_overlay(database)

            profiles = get_profiles(os.environ.get("profile_path"))

            for i in range(0, data[1]):
                print(f'skeleton_{i:03d}', flush=True)
                skeletons[f'skeleton_{i:03d}'] = deque(maxlen=1000)
        elif state.value is MState.END_MONITORING.value:
            break
        elif state.value is MState.APPEND_MONITORING.value:
            skeletons_frame = data[1]
            for i, sk_ind in enumerate(skeletons_frame):
                if sk_ind is not None:
                    skeletons[f'skeleton_{i:03d}'].append(sk_ind)
        elif state.value is MState.GET_PERSON_ID.value:
            logger.info("Getting person ID")
            ids = ['NoID'] * len(skeletons)
            for i, sk_ind in enumerate(skeletons):
                if len(skeletons[sk_ind]) > 10:
                    video = height_correction(list(map(lambda x:
                                                       list(map(lambda y: y.tolist(), x.player_coordinates)),
                                                       skeletons[sk_ind])))
                    video = align_on_the_floor(video)
                    video = skeleton_overlay(video)
                    print(len(video))
                    dm = basic_assembly_metric(database, video)
                    profile = np.median(dm, axis=1)
                    ids[i] = get_person_id(profile, profiles)
            qm.put(ids)
            break


def get_person_id(cur, stored):
    train_x, train_y = stored
    return knn(np.asarray(train_x, dtype=np.float64), np.asarray(train_y),
               np.asarray(cur, dtype=np.float64), 2)


def count_max(classes):
    counter = {}
    for e in classes:
        if e in counter:
            counter[e] += 1
        else:
            counter[e] = 1
    return max(counter, key=counter.get)


def get_profiles(filename):
    with open(os.environ.get("profiles_path"), "r+") as file:
        profiles = []
        profile_names = []
        file.readline()
        for line in file:
            profile_names.append(line.split(",")[:1])
            profiles.append(line.split(',')[2:])
    return profiles, profile_names


def knn(train_sample, train_answers, new_object, k, metric='euclidean'):
    nn = np.argsort(cdist(train_sample, np.expand_dims(new_object, axis=1).T, metric=metric).T, axis=1)[:, :k]
    prob_classes = train_answers[nn]
    # print(prob_classes)
    return count_max(prob_classes.reshape([len(nn[0])]))


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


def height_correction(sample_video):
    if len(sample_video) == 32:
        height = 1698
    else:
        height = 1900
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
    def skeletal_distance(s1, s2):
        res = 0
        for x, y in zip(s1, s2):
            res += distance.euclidean(x[:-1], y[:-1])
        return res / 17.

    distance_matrix = np.zeros(shape=(32, 0))
    for skeleton_frame in video_sequence:
        d = np.asarray(list(map(lambda ba: skeletal_distance(ba, skeleton_frame), basic_assembly)))
        d = np.expand_dims(d.T, axis=1)
        distance_matrix = np.hstack([distance_matrix, d])
    return distance_matrix