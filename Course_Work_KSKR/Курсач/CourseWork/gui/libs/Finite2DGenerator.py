from libs.TriangleStiffness import *
from libs.StiffnessData import StiffnessData
import scipy.linalg as sl
import matplotlib.tri as mtri


class Finite2DGenerator:
    def __init__(self, data: StiffnessData):
        self.dimension = 2
        self.data = data
        self.sorted_nodes = data.nodes
        self.stiffness_elements = self._extract_triangles(self.data.get_raw_nodes(), self.data.tri.simplices)
        self.m_s = len(self.sorted_nodes) * self.dimension
        self.stiffness_matrix = None
        self.calculated_loads = None
        self.calculated_moving = None
        self.moving_indexes = None
        self.__calc_load = None
        self.D, self.h = self.data.d(), self.data.thickness

    def get_stiffness_matrix(self):
        col_size = self.m_s // len(self.sorted_nodes)
        row_num = 0
        self.__calc_load = set()
        self.stiffness_matrix = np.zeros((self.m_s, self.m_s), dtype=np.float64)
        for node in self.sorted_nodes:
            elements = [e for e in self.stiffness_elements if node in e.nodes]
            indexes = [e.nodes.index(node) for e in elements]
            self._append_to_matrix(elements, indexes, row_num, col_size)
            row_num += self.dimension
        return self.stiffness_matrix

    def calculate_moving(self):
        g_matrix = self.get_stiffness_matrix()
        indexes = []
        index = 0
        for p, m in zip(self.data.loads, self.data.moving):
            if np.isnan(p[0]) and m[0] == 0:
                indexes.append(index)
            elif np.isnan(p[0]) or m[0] == 0:
                raise ValueError("p = " + str(p) + " m = " + str(m))
            index += 1
        a = np.delete(np.delete(g_matrix, indexes, 0), indexes, 1)
        b = np.delete(self.data.loads, indexes).reshape((-1, 1))
        self.calculated_moving = self.data.moving.copy()
        to_paste = [i for i in range(len(self.data.moving)) if i not in indexes]
        x = sl.solve(a, b)
        for i, v in zip(to_paste, x):
            self.calculated_moving[i] = v
        self.moving_indexes = to_paste
        return self.calculated_moving

    def _append_to_matrix(self, elements, indexes, r, col_size):
        col = 0
        col_nums = [0 for _ in indexes]
        rows = [e.get_stiffness_matrix(self.D, self.h)[i*col_size:i*col_size+col_size] for e, i in zip(elements, indexes)]
        for node in self.sorted_nodes:
            temp = np.zeros((col_size, col_size))
            for i in self._get_node_in_element_indexes(node, elements):
                temp += rows[i][:, col_nums[i]:col_nums[i] + col_size]
                col_nums[i] += col_size
            self.stiffness_matrix[r: r + self.dimension, col: col + col_size] += temp
            col += col_size

    def calculate_stress(self):
        col_size = self.m_s // len(self.sorted_nodes)
        row_num = 0
        stress = []
        for node in self.sorted_nodes:
            elements = [e for e in self.stiffness_elements if node in e.nodes]
            indexes = [e.nodes.index(node) for e in elements]
            self._stress_append(elements, indexes, stress, col_size)
            row_num += self.dimension
        return stress

    def _stress_append(self, elements, indexes, arr, col_size):
        col = 0
        col_nums = [0 for _ in indexes]
        rows = [e.get_stiffness_matrix(self.D, self.h)[i * col_size:i * col_size + col_size] for e, i in
                zip(elements, indexes)]
        cc = 0
        for node in self.sorted_nodes:
            temp = np.zeros((col_size, col_size))
            for i in self._get_node_in_element_indexes(node, elements):
                temp += rows[i][:, col_nums[i]:col_nums[i] + col_size] @ self.calculated_moving[cc: cc+2]
                col_nums[i] += col_size
                cc += 2
            cc = 0
            arr.append(temp)
            col += col_size

    @staticmethod
    def _get_node_in_element_indexes(node, elements):
        indexes = []
        i = 0
        for tr in elements:
            if node in tr.nodes:
                indexes.append(i)
            i += 1
        return indexes

    @staticmethod
    def _extract_triangles(nodes, simplices):
        triangles = []
        for coords in simplices:
            triangles.append(TriangleStiffness(nodes[coords[0]], nodes[coords[1]], nodes[coords[2]]))
        return triangles

    def draw(self, axis):
        axis.triplot(self.data.get_tri_x(), self.data.get_tri_y(), self.data.tri.simplices)
        axis.plot(self.data.get_tri_x(), self.data.get_tri_y(), 'o')

    def draw_displaced(self, axis):
        cn = self.data.get_raw_nodes()
        sc = []
        for n in cn:
            i = self.data.nodes.index(n)
            sc.append(np.abs(self.calculated_moving[2 * i][0]) + np.abs(self.calculated_moving[2 * i + 1][0]))
        triangulation = mtri.Triangulation(self.data.get_tri_x(), self.data.get_tri_y(), self.data.tri.simplices)
        axis.triplot(triangulation, '-k')
        axis.tricontourf(triangulation, sc)

