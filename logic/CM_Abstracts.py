# -*- coding: utf-8 -*-

from abc import abstractmethod

import pandas as pd
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QAbstractTableModel, Qt

from model.CM_classes import CognitiveModel
from model.CM_funcs import data_2_dbf


class AbstractLogic(QtWidgets.QWidget):
    """Abstract class for child forms"""
    modelChangedSignal = QtCore.pyqtSignal()

    def __init__(self, cognitive_model: CognitiveModel, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.cognitive_model = cognitive_model

    def notify_main(self):
        """Tell to main form, that model has changed"""
        self.modelChangedSignal.emit()

    @abstractmethod
    def model_changed(self):
        """When model changing, this method will update form"""
        pass


class pandasModel(QAbstractTableModel):
    """Model class for displaying pandas dataframe in QTableView
    Taken from https://gist.github.com/DataSolveProblems/972884bb9a53d5b2598e8674acc9e8ab"""
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[section]
        return None


class AbstractEstimatorLogic(AbstractLogic):
    """Abstract class for child forms to create data using pandas"""
    def __init__(self, regressions, dispersions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regressions = regressions
        self.dispersions = dispersions
        self.data = pd.DataFrame()
        self.table_model = pandasModel(self.data)

    @abstractmethod
    def model_changed(self):
        """When model changing, this method will update form"""
        pass

    def data_changed(self):
        self.table_model.model().layoutChanged.emit()

    def save_data(self):
        filename, filetype = QtWidgets.QFileDialog.getSaveFileName(parent=self,
                                                                   filter="Comma-separated values (*.csv);;"
                                                                          "Книга Excel (*.xlsx);;"
                                                                          "Data Base File (*.dbf);;"
                                                                          "JavaScript Object Notation (*.json)")
        try:
            if filetype == "Comma-separated values(*.csv)":
                if filename[-4:] != '.csv':
                    filename = filename + '.csv'
                self.data.to_csv(filename, index=False)
            elif filetype == "Книга Excel (*.xlsx)":
                if filename[-5:] != '.xlsx':
                    filename = filename + '.txt'
                self.data.to_excel(filename, index=False)
            elif filetype == "Data Base File (*.dbf)":
                if filename[-4:] != '.dbf':
                    filename = filename + '.dbf'
                data_2_dbf(self.data, self.cognitive_model.get_factors(), filename)
            elif filetype == "JavaScript Object Notation (*.json)":
                if filename[-5:] != '.json':
                    filename = filename + '.json'
                self.data.to_json(filename, orient='split', index=False, indent=4)
            else:
                return
        except OSError:
            QtWidgets.QMessageBox.critical(self,
                                           "Ошибка сохранения",
                                           "Не удалось сохранить данные в указанный файл: %s" % filename,
                                           buttons=QtWidgets.QMessageBox.Ok)
