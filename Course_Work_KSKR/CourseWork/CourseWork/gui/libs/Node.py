from shapely.geometry import Point
from math import ceil


class Node(Point):
    def __init__(self, *args):
        super().__init__(*args)
        self.moving = True
        self.order = None
        self.num = None

    def set_fixed(self):
        self.moving = False

    def is_moved(self):
        return self.moving

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()

    def __hash__(self):
        return ceil(31 * self.x + self.y)
