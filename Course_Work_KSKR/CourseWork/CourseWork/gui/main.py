import sys
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QApplication, QMainWindow
from design import Ui_MainWindow

from libs.Finite2DGenerator import Finite2DGenerator
from libs.StiffnessData import StiffnessData
from libs.Node import Node

default_bracket = [[500, 500], [1100, 500], [1400, 800], [1100, 1100], [500, 1100], [200, 800]]


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.figure = Figure()
        self.figureCanvas = FigureCanvas(self.figure)
        self.plotToolbar = NavigationToolbar(self.figureCanvas, self)
        self.data = StiffnessData()

    def setup_ui(self):
        self.ui.plotLayout.addWidget(self.figureCanvas)
        self.ui.plotLayout.addWidget(self.plotToolbar)
        self.setup_handlers()

    def setup_handlers(self):
        self.ui.setupBraketParametersBtn.clicked.connect(self.bracket_parameters_setup)
        self.ui.meshGenerateBtn.clicked.connect(self.mesh_parameters_setup)
        self.ui.setupLFparametersBtn.clicked.connect(self.load_fix_parameters_setup)
        self.ui.loadTypeBox.currentIndexChanged.connect(self.load_fix_box_type_changed)
        self.ui.fixTypeBox.currentIndexChanged.connect(self.load_fix_box_type_changed)

    def draw_plot(self):
        axis = self.figure.add_subplot(121) if len(self.figure.get_axes()) == 0 else self.figure.get_axes()[0]
        bracket = default_bracket.copy()
        bracket.append(bracket[0])
        bracket = np.array(bracket)
        axis.plot(bracket[:, 0], bracket[:, 1], c="blue")

    def draw_mesh(self):
        x = self.data.get_tri_x()
        y = self.data.get_tri_y()
        if x is not None and y is not None:
            axis = self.figure.get_axes()[0] if len(self.figure.get_axes()) > 0 else self.figure.add_subplot(121)
            axis.triplot(x, y, self.data.tri.simplices, c="lime")
            axis.plot(x, y, 'o')
            self.figure.canvas.draw()

    def calculate(self, p):
        if self.data.is_prepared():
            self.data.set_fixing(Node(default_bracket[0]))
            self.data.set_fixing(Node(default_bracket[1]))
            self.data.set_load(Node(default_bracket[2]), p)
            fg = Finite2DGenerator(self.data)
            moving = fg.calculate_moving()
            self.ui.resultText.setText(f"0 - {min(moving)}")
            fg.draw_displaced(self.figure.add_subplot(122)
                              if len(self.figure.get_axes()) < 2
                              else self.figure.get_axes()[1])
            self.figure.canvas.draw()
        else:
            self.show_popup("Ошибка", "Не введены необходимые данные")

    def bracket_parameters_setup(self):
        try:
            y = float(self.ui.youngaParameter.text())
            m = float(self.ui.poissonParameter.text())
            h = float(self.ui.thicknessParameter.text())
            self.ui.bracketParametersText.setText(f"""Параметры кронштейна:
Коэффициент Юнга: {y}
Коэффициент Пуассона: {m}
Толщина кронштейна: {h} мм""")
            self.data.poisson, self.data.young, self.data.thickness = m, y, float(h) / 1000.0
            self.draw_plot()
            self.ui.pageMesh.setEnabled(True)
        except ValueError:
            self.show_popup("Ошибка ввода", "Неккоректное число")

    def mesh_parameters_setup(self):
        try:
            size = float(self.ui.meshSizeInput.text())
            self.data.define_nodes_mesh(default_bracket.copy(), size)
            self.draw_mesh()
            self.setup_node_fix_load_info()
            self.set_checkable_boxes()
            val = float(self.ui.powerValue.text())
            self.calculate(val)
        except ValueError as e:
            self.show_popup("Ошибка ввода", "Неккоректное число", QMessageBox.Critical, None, f"{e}\n{e}")

    def load_fix_parameters_setup(self):
        fix = self.ui.fixParametersBox.findChildren(QtWidgets.QCheckBox)
        load = self.ui.loadParametersBox.findChildren(QtWidgets.QCheckBox)
        fix_indexes = []
        load_indexes = []
        i = 0
        for f, l in zip(fix, load):
            if f.isEnabled() and f.isChecked():
                fix_indexes.append(self.define_node_edge(f.text(), i))
            if l.isEnabled() and l.isChecked():
                load_indexes.append(self.define_node_edge(l.text(), i))
            i += 1
        if self.check_fix_load_validate(fix_indexes, load_indexes):
            print(fix_indexes, load_indexes)

    @staticmethod
    def define_node_edge(text, default_value):
        for i in default_bracket:
            if text == f"{i}":
                return i
        return default_value

    def check_fix_load_validate(self, fix, load):
        valid = True
        if len(fix) == 0 or len(load) == 0:
            self.show_popup("Неверные данные", "Не заданы закрепления или фиксация", QMessageBox.Warning)
            valid = False
        elif len(fix) + len(load) > 3:
            self.show_popup("Неверные данные", "Закрепления и фиксация заданны не корректно", QMessageBox.Warning)
            valid = False
        elif len([e for e in fix if e in load]) > 0:
            self.show_popup("Неверные данные", "Закрепления и фиксация совпадают", QMessageBox.Warning)
            valid = False
        return valid

    def load_fix_box_type_changed(self):
        self.set_checkable_boxes()

    def set_checkable_boxes(self):
        if self.ui.fixTypeBox.currentIndex() == 0:
            self.enable_fix_node_checkboxes()
        else:
            self.enable_fix_node_checkboxes(False)
        if self.ui.loadTypeBox.currentIndex() == 0:
            self.enable_load_node_checkboxes()
        else:
            self.enable_load_node_checkboxes(False)

    def setup_node_fix_load_info(self):
        fix = self.ui.fixParametersBox.findChildren(QtWidgets.QCheckBox)[0:3]
        load = self.ui.loadParametersBox.findChildren(QtWidgets.QCheckBox)[0:3]
        index = 0
        for f, l in zip(fix, load):
            f.setText(f"{default_bracket[index]}")
            l.setText(f"{default_bracket[index]}")
            index += 1
        self.ui.bottomFixNodeCheckBox.setChecked(True)
        self.ui.leftFixNodeCheckBox.setChecked(True)
        self.ui.rightLoadNodeCheckBox.setChecked(True)

    def enable_fix_node_checkboxes(self, enable=True):
        self.enable_checkboxes(self.ui.fixParametersBox, 0, 3, enable)
        self.enable_checkboxes(self.ui.fixParametersBox, 3, 6, not enable)

    def enable_load_node_checkboxes(self, enable=True):
        self.enable_checkboxes(self.ui.loadParametersBox, 0, 3, enable)
        self.enable_checkboxes(self.ui.loadParametersBox, 3, 6, not enable)

    @staticmethod
    def enable_checkboxes(widget, i, j, enable):
        for w in widget.findChildren(QtWidgets.QCheckBox)[i:j]:
            w.setCheckable(enable)
            w.setEnabled(enable)

    @staticmethod
    def show_popup(title, text, icon=QMessageBox.Critical, info=None, details=None):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setIcon(icon)
        if info is not None:
            msg.setInformativeText(info)
        if details is not None:
            msg.setDetailedText(details)
        return msg.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.setup_ui()
    win.show()
    sys.exit(app.exec_())
