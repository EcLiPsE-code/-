from scipy.spatial import Delaunay
from math import sqrt, ceil
from libs.Node import Node
from libs.TriangleStiffness import clockwise_angle_and_distance, set_clockwise_origin

import numpy as np


def split_via_delaunay(points, max_length):
    tri = Delaunay(points)
    # get set of edges from the simpleces
    edges = set()
    for simplex in tri.simplices:
        # simplex is one triangle: [ 4  5 17]
        edges.add((simplex[0], simplex[1]))
        edges.add((simplex[1], simplex[2]))
        edges.add((simplex[0], simplex[2]))
    # check if all edges are small enough
    # and add new points if not
    is_finished = True
    for edge in edges:
        p1, p2 = edge
        [x1, y1] = points[p1]
        [x2, y2] = points[p2]
        length = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
        if length > max_length:
            is_finished = False
            # split in how many pieces?
            n_pieces = ceil(length / max_length)
            for piece in range(1, int(n_pieces)):
                points.append([x1 + piece / float(n_pieces) * (x2 - x1), y1 + piece / float(n_pieces) * (y2 - y1)])
    if not is_finished:
        split_via_delaunay(points, max_length)


class StiffnessData:
    def __init__(self):
        self.loads = None
        self.moving = None
        self.points = None
        self.nodes = None
        self.poisson = None
        self.thickness = None
        self.young = None
        self.tri = None
        self.__nodes = None
        self.D = None

    def is_prepared(self):
        return self.nodes is not None and self.thickness is not None    \
               and self.young is not None and self.poisson is not None

    def d(self):
        if self.D is None:
            if self.is_prepared():
                hooke = np.array([
                    [1, self.poisson, 0],
                    [self.poisson, 1, 0],
                    [0, 0, (1 - self.poisson) / 2]
                ])
                self.D = (self.young / (1 - self.poisson ** 2)) * hooke
        return self.D

    def define_nodes_mesh(self, _points, size=None, split=True):
        if split:
            split_via_delaunay(_points, size)
        self.tri = Delaunay(_points)
        self.create_nodes(_points)
        self.points = _points
        self.loads = np.zeros((2 * len(self.nodes), 1))
        self.moving = np.ones((2 * len(self.nodes), 1))

    def define_nodes_mesh_by_parts(self, size, *args):
        result = []
        for arr in args:
            temp = arr.copy()
            split_via_delaunay(temp, size)
            result += temp
        self.define_nodes_mesh(result, None, False)
        self.clear_mesh(Node(400, 500))

    def clear_mesh(self, edge):
        temp = self.tri.simplices.tolist()
        on_edge = []
        for s in temp:
            if any([True for i in s if self.__nodes[i].y == edge.y and self.__nodes[i].x > edge.x]):
                on_edge.append(s)
        to_del = []
        for n in on_edge:
            pp = None
            for i in n:
                if self.__nodes[i].y == edge.y and self.__nodes[i].x > edge.x:
                    pp = self.__nodes[i]
            if pp is not None:
                if any(True for i in n if self.__nodes[i].y < pp.y):
                    to_del.append(n)
        for d in to_del:
            temp.remove(d)
        self.tri.simplices = np.array(temp)

    def set_fixing(self, point):
        index = self.nodes.index(point)
        self.moving[2 * index], self.moving[2 * index + 1] = 0, 0
        self.set_undefined_load(point)

    # set loads by Ox and Oy
    def set_load(self, point, value):
        index = self.nodes.index(point)
        self.loads[2 * index][0], self.loads[2 * index + 1][0] = value * np.cos(np.radians(55)), value * np.cos(np.radians(145))

    def set_undefined_load(self, point):
        self.set_load(point, np.nan)

    def get_tri_x(self):
        return [n.x for n in self.__nodes] if self.tri is not None else None

    def get_tri_y(self):
        return [n.y for n in self.__nodes] if self.tri is not None else None

    def create_nodes(self, points):
        self.__nodes = [Node(p) for p in points]
        duplicated_points = self._find_duplicated_elements(self.__nodes)
        for k in duplicated_points:  # change duplicated to one object
            for i in duplicated_points[k]:
                self.__nodes[i] = self.__nodes[k]
        # self.nodes = sorted(list(set(self.__nodes)), key=lambda point: [point.y, point.x])
        set_clockwise_origin([0, 0])  # TODO: fake
        self.nodes = StiffnessData.mesh_sort(self.__nodes)
        for i in range(len(self.nodes)):
            self.nodes[i].num = i

    def get_raw_nodes(self):
        return self.__nodes

    @staticmethod
    def _find_duplicated_elements(array):
        excluded = []  # for indexes which already appended to dictionary
        indexes = {}
        for i in range(len(array) - 1):
            if i in excluded:
                continue
            for j in range(i + 1, len(array)):
                if array[i] == array[j]:
                    if i in indexes:
                        indexes[i].append(j)
                    else:
                        indexes[i] = [j]
                    excluded.append(j)
        return indexes

    @staticmethod
    def mesh_sort(nds):
        nodes = sorted(list(set(nds)), key=clockwise_angle_and_distance)
        nn = nodes[0]
        nodes.remove(nn)
        nodes.append(nn)
        nodes.reverse()
        return nodes
