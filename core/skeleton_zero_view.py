import numpy as np


def T(X, tx, ty, tz):
    X = np.hstack([X, np.ones(shape=[X.shape[0], 1])])
    T = np.asarray([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
    ])
    return np.dot(T, X.T).T[:, :-1]


def R(X, rx=1, ry=1, rz=1):
    X = np.hstack([X, np.ones(shape=[X.shape[0], 1])])
    reflection_matrix = np.asarray([
        [rx, 0, 0, 0],
        [0, ry, 0, 0],
        [0, 0, rz, 0],
        [0, 0, 0, 1],
    ])
    return np.dot(reflection_matrix, X.T).T[:, :-1]


def M(X, phi=0, sign=1):
    X = np.hstack([X, np.ones(shape=[X.shape[0], 1])])
    rotate_matrix = np.asarray([
        [np.cos(phi), 0, -sign * np.sin(phi), 0],
        [0, 1, 0, 0],
        [sign * np.sin(phi), 0, np.cos(phi), 0],
        [0, 0, 0, 1]
    ])
    return np.dot(rotate_matrix, X.T).T[:, :-1]


def vec_prod(v1, v2):
    n = np.array([
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[0] * v2[2] - v1[2] * v2[0],
        v1[0] * v2[1] - v1[1] * v2[0],
    ])
    return n


def get_angle(point1, point2, measure="rad"):
    cosa = np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2) + 1e-10)
    alpha = np.arccos(cosa)
    if measure == "rad":
        return alpha
    elif measure == "dg":
        return (alpha * 180) / np.pi
    else:
        return None


def swap(s, i, j):
    temp = s[i].copy()
    s[i] = s[j].copy()
    s[j] = temp
    return s


def shoulder_correction(skeletal_model, is_ref=1):
    center = skeletal_model[0].copy()

    skeletal_model = T(skeletal_model, *(-center))
    if is_ref > 0:
        skeletal_model = M(skeletal_model, np.pi)
    skeletal_model = T(skeletal_model, *center)

    center = skeletal_model[4].copy()
    skeletal_model = T(skeletal_model, *(-center))
    j1, j2 = skeletal_model[[4, 7], :]
    a, b, c = j2 - j1
    p1 = np.array([a, 0, c])
    p2 = np.array([a, 0, 0])
    skeletal_model = M(skeletal_model,
                       get_angle(p1, p2, 'rad'),
                       sign=np.sign(-a * c)
                       )
    skeletal_model = T(skeletal_model, *center)
    return skeletal_model


def get_normal_vector_shoulder_head(s):
    j1, j2 = s[[16, 2], :]  # head
    v1 = j2 - j1
    j1, j2 = s[[16, 7], :]  # left
    v2 = j2 - j1
    try:
        a1 = np.cross(v1, v2)
        a2 = np.linalg.norm(a1)
        print(a1, a2, flush=True)
        res = a1 / a2
    except ZeroDivisionError:
        res = np.ones(shape=(3,))
    return res


def need_correction(v1, v2):
    a, b, c = v1
    v1 = np.asarray([a, 0, c])
    a, b, c = v2
    v2 = np.asarray([a, 0, c])
    try:
        v3 = np.sign(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
    except ZeroDivisionError:
        v3 = np.ones(shape=(3,))
    return v1, v2, v3


def get_move_direction(s1, s2):
    v_0 = s2[[0], :] - s1[[0], :]
    v_2 = s2[[2], :] - s1[[2], :]
    return np.mean(np.vstack((v_0, v_2)), axis=0)


def skeleton_repair(s):
    swap(s, 4, 7)
    swap(s, 5, 8)
    swap(s, 6, 9)
    swap(s, 10, 13)
    swap(s, 11, 14)
    swap(s, 12, 15)
    return s


def get_v(x):
    return "Right" if x > 0 else "Fail"


def skeleton_stability(sk_prev, sk, smoothed_move=None, alpha=0.9):
    #  Вычислить нормальный вектор между тремя точками - голова - шея - плечо
    hs_normal = get_normal_vector_shoulder_head(sk)

    #  Вычислить вектор направления движения
    move_direction = get_move_direction(sk_prev, sk)

    #  Сглаживание вектора направления движения
    smoothed_move_updated = (move_direction
                             if smoothed_move is None else
                             alpha * smoothed_move + (1 - alpha) * move_direction)

    #  Определить необходимость перестановки координат в случае ошибки kinect
    _, _, ret = need_correction(hs_normal,
                                smoothed_move_updated.ravel())

    #  Если необходимо (ret < 0) переставить координаты местами
    if ret < 0:
        sk = skeleton_repair(sk.copy())

    #  Получить новый нормальный вектор между тремя точками голова-шея-плечо
    hs_normal = get_normal_vector_shoulder_head(sk)

    #  Развернуть скелетную модель лицом к камере
    sk = shoulder_correction(sk, np.sign(hs_normal[2]))

    return sk, smoothed_move_updated


def skeleton_correction(skeletal_video):
    alpha = 0.9
    smoothed = None
    processed_data = []
    for i in range(1, len(skeletal_video)):
        skeletal_model, smoothed = skeleton_stability(skeletal_video[i - 1], skeletal_video[i], smoothed, alpha=alpha)
        processed_data.append(skeletal_model)
    return processed_data


