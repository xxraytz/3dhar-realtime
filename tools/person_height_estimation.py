import numpy as np
import math


def shorty(func):
    def wrapper(self, *args, ):
        x, y = args[0][:2], args[1][:2]
        return func(self, x, y)

    return wrapper


class PersonHeight:
    def __init__(self, id, sample):
        self.id = id
        self.sample = sample

    def get_heights(self, n_max_heights, conf=None):
        if conf:
            hei = [self.__person_height(skeleton) if (not (-1 in c)) else 0 for (c, skeleton) in zip(conf, self.sample)]
        else:
            hei = [self.__person_height(skeleton) for skeleton in self.sample]
        return sorted(hei, reverse=True)[:n_max_heights]

    @staticmethod
    def __person_height(person):
        key_points = [
            [12, 11],
            [11, 10],
            [10, 0],
            [0, 1],
            [0, 1],
            [1, 16],
            [1, 16],
            [16, 2],
            [16, 2],
            [2, 3],
            [2, 3],
            [15, 4],
            [14, 13],
            [13, 0],
        ]

        distance = sum([math.sqrt(np.sum((person[a] - person[b]) ** 2)) for a, b in key_points]) / 2.

        return distance
