# -*- coding: utf-8 -*-
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from forms.CM_ProcessForm import Ui_ProcessForm
from logic.CM_Abstracts import AbstractEstimatorLogic
from model.CM_funcs import estimate_transient_response, normalize


class PlotDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, data=pd.DataFrame()):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("График переходного процесса")
        self.resize(600, 600)
        self.mainBox = QtWidgets.QVBoxLayout()

        self.fig = plt.figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.mainBox.addWidget(self.toolbar)
        self.mainBox.addWidget(self.canvas)

        self.box = QtWidgets.QDialogButtonBox(QtCore.Qt.Horizontal)
        self.btnOK = QtWidgets.QPushButton("&OK")
        self.box.addButton(self.btnOK, QtWidgets.QDialogButtonBox.AcceptRole)
        self.box.accepted.connect(self.accept)

        self.mainBox.addWidget(self.box)
        self.setLayout(self.mainBox)
        self.ax = self.fig.add_subplot()
        self.ax.plot(data, label=data.columns)
        self.ax.legend()
        self.ax.grid()
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.canvas.draw()
        self.update()


class ProcessWindow(AbstractEstimatorLogic, Ui_ProcessForm):

    class ProcessThread(AbstractEstimatorLogic.CalculatingThread):
        def run(self):
            data = estimate_transient_response(*self.args, **self.kwargs)
            self.sendDataSignal.emit(data)

    def __init__(self, params_dict, cognitive_model):
        super(ProcessWindow, self).__init__(params_dict, cognitive_model)
        self.setupUi(self)
        self.btnSave.clicked.connect(self.save_data)
        self.btnCreate.clicked.connect(self.click_create)
        self.btnPlot.clicked.connect(self.click_plot)
        self.model_changed()
        self.calculating_thread = self.ProcessThread()
        self.calculating_thread.sendDataSignal.connect(self.get_data_from_thread)

    def model_changed(self):
        self.data = pd.DataFrame()
        self.data_changed()
        factors = self.cognitive_model.get_factors()
        n = len(factors)
        if not n:
            self.setDisabled(True)
            self.tableWidget.setColumnCount(0)
            return
        elif max([f.scale for f in factors]) > 0:
            QtWidgets.QMessageBox.information(self,
                                              'Ошибка факторов',
                                              'В модели обнаружены качественные факторы\n'
                                              'Анализ переходного процесса невозможен',
                                              buttons=QtWidgets.QMessageBox.Ok)
            self.setDisabled(True)
            self.tableWidget.setColumnCount(0)
            return
        else:
            self.setEnabled(True)
        self.tableWidget.setColumnCount(n)
        for j in range(n):
            item = QtWidgets.QTableWidgetItem(factors[j].name)
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.tableWidget.setHorizontalHeaderItem(j, item)
            max_value = factors[j].max_value
            min_value = factors[j].min_value
            value = (max_value + min_value) * 0.5
            for i in (0, 1):
                spin = QtWidgets.QDoubleSpinBox()
                spin.setMaximum(max_value)
                spin.setMinimum(min_value)
                spin.setValue(value)
                self.tableWidget.setCellWidget(i, j, spin)
        self.tableWidget.resizeColumnsToContents()

    def data_changed(self):
        super().data_changed()
        if not self.data.shape[0]:
            self.btnPlot.setDisabled(True)
        else:
            self.btnPlot.setEnabled(True)

    def click_create(self):
        self.create_data()

    def create_data(self):
        self.data = pd.DataFrame()
        self.data_changed()
        self.check_parameters()
        factors = self.cognitive_model.get_factors()
        n = len(factors)
        state_0 = {factors[j].name: self.tableWidget.cellWidget(0, j).value() for j in range(n)}
        state_1 = {factors[j].name: self.tableWidget.cellWidget(1, j).value() for j in range(n)}
        if len(factors) < 1:
            return
        self.busySignal.emit(True)
        self.calculating_thread.start(self.params_dict['regressions'], factors, state_0, state_1)

    def click_plot(self):
        factors = self.cognitive_model.get_factors()
        data = pd.DataFrame()
        for f in factors:
            data[f.name] = self.data[f.name].apply(lambda x: normalize(x, f))
        dialog = PlotDialog(self, data)
        dialog.exec()
