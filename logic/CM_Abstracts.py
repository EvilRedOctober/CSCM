# -*- coding: utf-8 -*-
"""Contains abstract classes for PyQt forms"""

from abc import abstractmethod

import pandas as pd
from PyQt5 import QtCore, QtWidgets

from model.CM_classes import CognitiveModel
from model.CM_funcs import data_2_dbf, make_imitation_data


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


class pandasModel(QtCore.QAbstractTableModel):
    """Model class for displaying pandas dataframe in QTableView
    Taken from https://gist.github.com/DataSolveProblems/972884bb9a53d5b2598e8674acc9e8ab"""

    def __init__(self, data):
        QtCore.QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[section]
        return None


class AbstractEstimatorLogic(AbstractLogic):
    """Abstract class for child forms to create data using pandas library.
    Working with linear regressions parameters"""
    needParametersSignal = QtCore.pyqtSignal()
    busySignal = QtCore.pyqtSignal(bool)

    class CalculatingThread(QtCore.QThread):
        sendDataSignal = QtCore.pyqtSignal(pd.DataFrame)

        def __init__(self, *args, **kwargs):
            QtCore.QThread.__init__(self)
            self.args = args
            self.kwargs = kwargs

        def start(self, *args, **kwargs):
            super().start()
            self.args = args
            self.kwargs = kwargs

        def run(self):
            data = make_imitation_data(*self.args, **self.kwargs)
            self.sendDataSignal.emit(data)

    def __init__(self, params_dict: dict, *args, **kwargs):
        """
        :param params_dict: dictionary of shape: {'regressions': np.ndarray, 'dispersions': np.ndarray,
        'visiting_order': list}
        """
        super().__init__(*args, **kwargs)
        self.params_dict = params_dict
        self.data = pd.DataFrame()
        self.table_model = pandasModel(self.data)
        self.calculating_thread = self.CalculatingThread()
        self.calculating_thread.sendDataSignal.connect(self.get_data_from_thread)

    @abstractmethod
    def model_changed(self):
        """When model changing, this method will update form"""
        pass

    def data_changed(self):
        self.tableView: QtWidgets.QTableView
        self.table_model = pandasModel(self.data)
        if not self.data.shape[0]:
            self.table_model = QtCore.QStringListModel()
            self.btnSave.setDisabled(True)
        else:
            self.btnSave.setEnabled(True)
        self.tableView.setModel(self.table_model)
        self.tableView.resizeColumnsToContents()

    def check_parameters(self):
        if len(self.params_dict.keys()) < 3:
            self.needParametersSignal.emit()

    @QtCore.pyqtSlot(pd.DataFrame)
    def get_data_from_thread(self, data: pd.DataFrame):
        self.data = data
        self.busySignal.emit(False)
        self.data_changed()

    def create_data(self, *args, **kwargs):
        self.data = pd.DataFrame()
        self.data_changed()
        self.check_parameters()
        factors = self.cognitive_model.get_factors()
        if len(factors) < 1:
            return
        self.busySignal.emit(True)
        self.calculating_thread.start(self.params_dict['regressions'], self.params_dict['dispersions'], factors,
                                      self.params_dict['visiting_order'], *args, **kwargs)

    def save_data(self):
        filename, filetype = QtWidgets.QFileDialog.getSaveFileName(parent=self,
                                                                   filter="Comma-separated values (*.csv);;"
                                                                          "Книга Excel (*.xlsx);;"
                                                                          "Data Base File (*.dbf);;"
                                                                          "JavaScript Object Notation (*.json)")
        try:
            if filetype == "Comma-separated values (*.csv)":
                if filename[-4:] != '.csv':
                    filename = filename + '.csv'
                self.data.to_csv(filename, index=False)
            elif filetype == "Книга Excel (*.xlsx)":
                if filename[-5:] != '.xlsx':
                    filename = filename + '.xlsx'
                self.data.to_excel(filename, index=False)
            elif filetype == "Data Base File (*.dbf)":
                if filename[-4:] != '.dbf':
                    filename = filename + '.dbf'
                data_2_dbf(self.data, self.cognitive_model.get_factors(), filename)
            elif filetype == "JavaScript Object Notation (*.json)":
                if filename[-5:] != '.json':
                    filename = filename + '.json'
                self.data.to_json(filename, orient='split', index=False, indent=4)
        except OSError:
            QtWidgets.QMessageBox.critical(self,
                                           "Ошибка сохранения",
                                           "Не удалось сохранить данные в указанный файл: %s" % filename,
                                           buttons=QtWidgets.QMessageBox.Ok)
