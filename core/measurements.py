import numpy as np
from scipy.spatial.distance import cdist, mahalanobis, euclidean


def knn(train_sample, train_answers, new_object, ro_values, k, metric='euclidean'):
    d = cdist(
        train_sample,
        np.expand_dims(new_object, axis=1).T,
        metric=metric
    ).T
    better_candidates = np.where(d < np.asarray([ro_values[name] for name in train_answers]))[1]
    d = d[0][better_candidates]
    train_answers = train_answers[better_candidates]
    k = k if k < d.shape[0] else d.shape[0]
    nn = np.sort(d)[:k]  # TODO: There no mistake
    nn_args = np.argsort(d)[:k]
    prob_classes = train_answers[nn_args]
    # print(prob_classes)
    return count_max(prob_classes)


def count_max(classes):
    counter = {}
    for e in classes:
        if e in counter:
            counter[e] += 1
        else:
            counter[e] = 1
    return max(counter, key=counter.get) if len(counter) > 0 else 'NoID'


def get_train_params(x, y, classes):
    center = {}
    Ss = {}
    ms = {}
    sko = {}
    for class_id in classes:
        curr_x = x[np.where(y == class_id)[0]]
        ms[class_id] = np.mean(curr_x, axis=0)
        sko[class_id] = np.std(curr_x, axis=0)
        center[class_id] = np.mean(
            curr_x,
            axis=0)
        curr_x = (curr_x - ms[class_id]) / sko[class_id]
        Ss[class_id] = np.cov(curr_x.T, rowvar=True)
    return ms, sko, Ss, center


def calc_measure(x_test, ms, sko, covs, center, class_names, measure='euclidean'):
    res = []
    values = None
    for x in x_test:
        if measure == 'mahalanobis':
            values = np.array([
                mahalanobis((x - ms[j]) / sko[j],
                            center[j],
                            np.linalg.inv(covs[j])) for j in class_names])
        elif measure == 'euclidean':
            values = np.array([
                euclidean((x - ms[j]) / sko[j], center[j]) for j in class_names
            ])
        res.append(class_names[np.argmin(values)])
    return most_common(res)


def most_common(lst):
    return max(set(lst), key=lst.count) if len(lst) > 0 else 'NoID'
