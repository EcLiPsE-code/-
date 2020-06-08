from abc import ABC

from libs.Node import *
from shapely.geometry import Polygon
import numpy as np
import math
import operator
from functools import reduce


origin = []
refvec = [0, 1]


def set_clockwise_origin(value_list):
    global origin
    origin = value_list


def clockwise_angle_and_distance(point):
    vector = [point.x - origin[0], point.y - origin[1]]  # Vector between point and the origin: v = p - o
    lenvector = math.hypot(vector[0], vector[1])  # Length of vector: ||v||
    if lenvector == 0:  # If length is zero there is no angle
        return -math.pi, 0
    normalized = [vector[0] / lenvector, vector[1] / lenvector]  # Normalize vector: v/||v||
    dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1] * normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2 * math.pi + angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector


class TriangleStiffness(Polygon, ABC):
    def __init__(self, a: Node, b: Node, c: Node):
        super().__init__([a, b, c])
        self.a = a
        self.b = b
        self.c = c
        self.__define_nodes_order([a, b, c])
        self._stiffness = None

    def draw(self, axis):
        xy = np.array([self.a.coords[0], self.b.coords[0], self.c.coords[0], self.a.coords[0]])
        axis.plot(xy[:, 0], xy[:, 1])

    def get_stiffness_matrix(self, d, h):
        if self._stiffness is None:
            i, j, k = 0, 1, 2
            b = np.array([
                [self._nodes[j].y - self._nodes[k].y, 0, self._nodes[k].y - self._nodes[i].y, 0,
                 self._nodes[i].y - self._nodes[j].y, 0],
                [0, self._nodes[k].x - self._nodes[j].x, 0, self._nodes[i].x - self._nodes[k].x, 0,
                 self._nodes[j].x - self._nodes[i].x],
                [self._nodes[k].x - self._nodes[j].x, self._nodes[j].y - self._nodes[k].y,
                 self._nodes[i].x - self._nodes[k].x, self._nodes[k].y - self._nodes[i].y,
                 self._nodes[j].x - self._nodes[i].x, self._nodes[i].y - self._nodes[j].y],
            ], dtype=np.float64) / (2 * self.area)
            self._stiffness = self.area * h * b.transpose() @ d @ b
        return self._stiffness

    def __define_nodes_order(self, nodes):
        global origin
        start_node = min(nodes, key=lambda n: n.num)
        coords = [a.coords[0] for a in nodes]  #
        center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
        self._nodes = [Node(p) for p in sorted(coords, key=lambda coord: (-135 - math.degrees(
            math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360, reverse=True)]
        temp = self._nodes[:self._nodes.index(start_node)]
        for i in temp:
            self._nodes.remove(i)
            self._nodes.append(i)
        # nodes.remove(start_node)
        # origin = [start_node.x, start_node.y]
        # self._nodes = sorted(nodes, key=clockwiseangle_and_distance)
        # self._nodes.append(start_node)
        # self._nodes.reverse()

    @property
    def nodes(self):
        return self._nodes
