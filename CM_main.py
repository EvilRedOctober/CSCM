# -*- coding: utf-8 -*-

# TODO list:
#  Cognitive model class (CM)
#     1. Creation by using observations data
#     1.1. Correlation estimating between different scales
#     1.2. Loading data (pandas?) from files (.csv, .dbf, .json, .xlsx)
#     1.3. Matching columns to factors

import sys

from PyQt5 import QtWidgets

from logic.CM_MainLogic import MainWindow

app = QtWidgets.QApplication([])

app.setStyle("Fusion")
application = MainWindow()
application.show()

sys.exit(app.exec())
