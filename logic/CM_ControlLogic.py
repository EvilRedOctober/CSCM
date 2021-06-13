# -*- coding: utf-8 -*-
import pandas as pd
from PyQt5 import QtWidgets, QtCore

from forms.CM_ControlForm import Ui_ControlForm
from logic.CM_Abstracts import AbstractEstimatorLogic
from model.CM_funcs import estimate_prediction, gradient_method, criterion


class ControlWindow(AbstractEstimatorLogic, Ui_ControlForm):

    def __init__(self, params_dict, cognitive_model):
        super(ControlWindow, self).__init__(params_dict, cognitive_model)
        self.setupUi(self)
        self.model_changed()
        self.btnSave.clicked.connect(self.save_data)
        self.btnCreate.clicked.connect(self.click_create)
        self.targets = {}
        self.weights = {}

    def model_changed(self):
        self.data = pd.DataFrame()
        self.data_changed()
        factors = self.cognitive_model.get_factors()
        n = len(factors)
        if not n:
            self.setDisabled(True)
            self.tableWidget.setColumnCount(0)
            return
        elif max([f.scale for f in factors]) > 1:
            QtWidgets.QMessageBox.information(self,
                                              'Ошибка факторов',
                                              'В модели обнаружены номинальные факторы\n'
                                              'Определение управления по модели невозможно',
                                              buttons=QtWidgets.QMessageBox.Ok)
            self.setDisabled(True)
            self.tableWidget.setColumnCount(0)
            self.tableTargeted.setColumnCount(0)
            return
        elif min([f.role for f in factors]) > 0:
            QtWidgets.QMessageBox.information(self,
                                              'Ошибка факторов',
                                              'В модели отсутствуют управляемые факторы\n'
                                              'Определение управления по модели невозможно',
                                              buttons=QtWidgets.QMessageBox.Ok)
            self.setDisabled(True)
            self.tableWidget.setColumnCount(0)
            self.tableTargeted.setColumnCount(0)
            return
        elif max([f.role for f in factors]) < 2:
            QtWidgets.QMessageBox.information(self,
                                              'Ошибка факторов',
                                              'В модели отсутствуют целевые факторы\n'
                                              'Определение управления по модели невозможно',
                                              buttons=QtWidgets.QMessageBox.Ok)
            self.setDisabled(True)
            self.tableWidget.setColumnCount(0)
            self.tableTargeted.setColumnCount(0)
            return
        else:
            self.setEnabled(True)
        self.tableWidget.setColumnCount(n)
        # First state
        for j in range(n):
            item = QtWidgets.QTableWidgetItem(factors[j].name)
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.tableWidget.setHorizontalHeaderItem(j, item)
            if factors[j].scale == 0:
                max_value = factors[j].max_value
                min_value = factors[j].min_value
                spin = QtWidgets.QDoubleSpinBox()
            else:
                max_value = factors[j].max_value
                min_value = 1
                spin = QtWidgets.QSpinBox()
            value = (max_value + min_value) * 0.5
            spin.setMaximum(max_value)
            spin.setMinimum(min_value)
            spin.setValue(value)
            self.tableWidget.setCellWidget(0, j, spin)
        # Targeted factors
        targeted = [f for f in factors if f.role == 3]
        self.tableTargeted.setColumnCount(len(targeted))
        for j, y in enumerate(targeted):
            item = QtWidgets.QTableWidgetItem(y.name)
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.tableTargeted.setHorizontalHeaderItem(j, item)
            if y.scale == 0:
                max_value = y.max_value
                min_value = y.min_value
                spin = QtWidgets.QDoubleSpinBox()
            else:
                max_value = y.max_value
                min_value = 1
                spin = QtWidgets.QSpinBox()
            value = (max_value + min_value) * 0.5
            spin.setMaximum(max_value)
            spin.setMinimum(min_value)
            spin.setValue(value)
            self.tableTargeted.setCellWidget(0, j, spin)
            spinWeight = QtWidgets.QSpinBox()
            spinWeight.setMinimum(1)
            spinWeight.setMaximum(99)
            self.tableTargeted.setCellWidget(1, j, spinWeight)
        self.tableWidget.resizeColumnsToContents()
        self.tableTargeted.resizeColumnsToContents()

    def data_changed(self):
        self.textResults.clear()
        super().data_changed()
        if self.data.shape[0]:
            self.textResults.append('Оценки значений факторов:')
            factors = self.cognitive_model.get_factors()
            ans = estimate_prediction(self.data, factors)
            for i, l in enumerate(ans):
                self.textResults.append(factors[i].name)
                self.textResults.append(str(l))
            E = criterion(factors, self.weights, self.targets, self.data)
            self.textResults.append('Оценки значения критерия близости E = %.3f' % E)

    def click_create(self):
        M = self.spinBox.value()
        factors = self.cognitive_model.get_factors()
        targeted = [f for f in factors if f.role == 3]
        n = len(factors)
        m = len(targeted)
        S = sum([self.tableTargeted.cellWidget(1, j).value() for j in range(m)])
        state = {factors[j].name: self.tableWidget.cellWidget(0, j).value() for j in range(n)}
        self.targets = {targeted[j].name: self.tableTargeted.cellWidget(0, j).value() for j in range(m)}
        self.weights = {targeted[j].name: self.tableTargeted.cellWidget(1, j).value()/S for j in range(m)}
        self.needParametersSignal.emit()
        recommended_state = gradient_method(self.params_dict['regressions'], factors, self.weights, self.targets, state)
        self.create_data(M=M, fixed_controls=True, state=recommended_state)
