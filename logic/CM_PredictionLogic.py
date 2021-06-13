# -*- coding: utf-8 -*-
import pandas as pd
from PyQt5 import QtWidgets, QtCore

from forms.CM_PredictionForm import Ui_PredictionForm
from logic.CM_Abstracts import AbstractEstimatorLogic
from model.CM_funcs import estimate_prediction


class PredictionWindow(AbstractEstimatorLogic, Ui_PredictionForm):

    def __init__(self, params_dict, cognitive_model):
        super(PredictionWindow, self).__init__(params_dict, cognitive_model)
        self.setupUi(self)
        self.model_changed()
        self.btnSave.clicked.connect(self.save_data)
        self.btnCreate.clicked.connect(self.click_create)

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
                                              'Предсказание по модели невозможно',
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
        self.tableWidget.resizeColumnsToContents()

    def data_changed(self):
        self.textResults.clear()
        super().data_changed()
        level = self.spinConfidence.value()
        if self.data.shape[0]:
            self.textResults.append('Оценки значений факторов:')
            factors = self.cognitive_model.get_factors()
            ans = estimate_prediction(self.data, factors, level)
            for i, l in enumerate(ans):
                self.textResults.append(factors[i].name)
                self.textResults.append(str(l))

    def click_create(self):
        M = self.spinBox.value()
        Fixed_observables = self.checkBox.isChecked()
        factors = self.cognitive_model.get_factors()
        n = len(factors)
        state = {factors[j].name: self.tableWidget.cellWidget(0, j).value() for j in range(n)}
        self.create_data(M=M, fixed_controls=True, fixed_observables=Fixed_observables, state=state)
