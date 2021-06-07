# -*- coding: utf-8 -*-

# TODO list:
#  Cognitive model class (CM)
#     5. Manually creation
#     6. Creation by using observations data
#  Way to save/open observation data/ imitation data to/from db (it could be .dbf, .db, .csv, others???)
#     0. just a table with names and values
#     2. Ways to load/save

import sys

from PyQt5 import QtWidgets

from logic.CM_MainLogic import MainWindow

app = QtWidgets.QApplication([])

app.setStyle("Fusion")
application = MainWindow()
application.show()

sys.exit(app.exec())
