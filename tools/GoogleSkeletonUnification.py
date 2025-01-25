import mediapipe as mp
import numpy as np

def convert_to_kinect_coords(mp_landmarks):
    if mp_landmarks is None:
        return

    uniform_landmarks = np.zeros((17, 3))

    Ch = 1.01
    Csx = 1.01
    Csy = 1.05

    uniform_landmarks[5] = np.array([mp_landmarks[14].x, mp_landmarks[14].y, mp_landmarks[14].z])
    uniform_landmarks[8] = np.array([mp_landmarks[13].x, mp_landmarks[13].y, mp_landmarks[13].z])
    uniform_landmarks[11] = np.array([mp_landmarks[26].x, mp_landmarks[26].y, mp_landmarks[26].z])
    uniform_landmarks[14] = np.array([mp_landmarks[25].x, mp_landmarks[25].y, mp_landmarks[25].z])
    uniform_landmarks[16] = (np.array([mp_landmarks[12].x, mp_landmarks[12].y, mp_landmarks[12].z]) +
                             np.array([mp_landmarks[11].x, mp_landmarks[11].y, mp_landmarks[11].z])) / 2
    uniform_landmarks[0] = (np.array([mp_landmarks[24].x, mp_landmarks[24].y, mp_landmarks[24].z]) +
                            np.array([mp_landmarks[23].x, mp_landmarks[23].y, mp_landmarks[23].z])) / 2

    uniform_landmarks[3] = np.zeros(3)
    for i in range(11):
        uniform_landmarks[3] += np.array([mp_landmarks[i].x, mp_landmarks[i].y, mp_landmarks[i].z])
    uniform_landmarks[3] /= 11

    uniform_landmarks[2] = (uniform_landmarks[3] + uniform_landmarks[16]) / 2
    uniform_landmarks[1] = (uniform_landmarks[0] + uniform_landmarks[16]) / 2
    uniform_landmarks[6] = np.array([mp_landmarks[16].x, mp_landmarks[16].y, mp_landmarks[16].z])
    uniform_landmarks[9] = np.array([mp_landmarks[15].x, mp_landmarks[15].y, mp_landmarks[15].z])
    uniform_landmarks[12] = np.array([mp_landmarks[28].x, mp_landmarks[28].y, mp_landmarks[28].z])
    uniform_landmarks[15] = np.array([mp_landmarks[27].x, mp_landmarks[27].y, mp_landmarks[27].z])

    uniform_landmarks[10] = np.array([mp_landmarks[24].x * Ch, mp_landmarks[24].y, mp_landmarks[24].z])
    uniform_landmarks[13] = np.array([mp_landmarks[23].x / Ch, mp_landmarks[23].y, mp_landmarks[23].z])

    uniform_landmarks[4] = np.array([mp_landmarks[12].x * Csx, mp_landmarks[12].y * Csy, mp_landmarks[12].z])
    uniform_landmarks[7] = np.array([mp_landmarks[11].x / Csx, mp_landmarks[11].y * Csy, mp_landmarks[11].z])

    return uniform_landmarks