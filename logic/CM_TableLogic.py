# -*- coding: utf-8 -*-

from PyQt5 import QtGui
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtCore import Qt

from forms.CM_TableForm import Ui_TableForm
from logic.CM_Abstracts import AbstractLogic
from model.CM_funcs import to_cheddoc
from model.CM_classes import SCALES, ROLES

HORIZONTAL_LABELS = ('Имя фактора', 'Шкала', 'Роль')
COLORS = (("#fff0f0", "#ffd2d2"),
          ("#deeaf6", "#bdd6ee"),
          ("#e2efd9", "#c8e0b3"))


class TableWindow(AbstractLogic, Ui_TableForm):

    def __init__(self, cognitive_model):
        super(TableWindow, self).__init__(cognitive_model)
        self.setupUi(self)
        self.model_changed()

    @staticmethod
    def getItem(name, flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled, color='#ffc800'):
        item = QTableWidgetItem(name)
        item.setFlags(flags)
        item.setBackground(QtGui.QBrush(QtGui.QColor(color)))
        item.setForeground(QtGui.QBrush(QtGui.QColor('#000000')))
        return item

    def model_changed(self):
        # Change the table
        # Preparing data
        matrix = self.cognitive_model.get_matrix_of_links()
        factors = self.cognitive_model.get_factors()
        list_id = [f.get_id() for f in factors]
        n = len(factors)

        table = self.tableWidget
        table.showGrid()
        # Reshaping and changing labels
        table.clear()
        table.setColumnCount(3 + n)
        table.setRowCount(n)
        # Adding headers

        for j in (0, 1, 2):
            table.setHorizontalHeaderItem(j, self.getItem(HORIZONTAL_LABELS[j]))
            table.horizontalHeaderItem(j).setTextAlignment(Qt.AlignHCenter)
        for i in range(n):
            table.setVerticalHeaderItem(i, self.getItem(list_id[i]))
            table.setHorizontalHeaderItem(i + 3, self.getItem(list_id[i]))
        # Inserting factors data
        for i, f in enumerate(factors):
            color1 = COLORS[f.scale][i % 2]
            table.setItem(i, 0, self.getItem(f.name, color=color1))
            table.setItem(i, 1, self.getItem(SCALES[f.scale], color=color1))
            table.setItem(i, 2, self.getItem(ROLES[f.role], color=color1))
            # Inserting matrix of links
            for j in range(n):
                if matrix[i][j]:
                    cell = '%.f%%, %s' % (matrix[i, j]*100, to_cheddoc(matrix[i, j]))
                    table.setItem(i, j + 3, self.getItem(cell, color="#f0f0f0"))
                else:
                    table.setItem(i, j + 3, self.getItem(None, color="#c8c8c8"))
        table.resizeColumnsToContents()
