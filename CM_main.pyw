# -*- coding: utf-8 -*-


import sys

from PyQt5 import QtWidgets

from logic.CM_MainLogic import MainWindow

app = QtWidgets.QApplication([])

app.setStyle("Fusion")
application = MainWindow()
application.show()

sys.exit(app.exec())
