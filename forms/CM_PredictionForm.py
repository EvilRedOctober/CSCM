# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CM_PredictionForm.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_PredictionForm(object):
    def setupUi(self, PredictionForm):
        PredictionForm.setObjectName("PredictionForm")
        PredictionForm.resize(620, 500)
        PredictionForm.setMinimumSize(QtCore.QSize(620, 500))
        font = QtGui.QFont()
        font.setFamily("Franklin Gothic Medium")
        font.setPointSize(10)
        PredictionForm.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/prediction.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        PredictionForm.setWindowIcon(icon)
        self.verticalLayout = QtWidgets.QVBoxLayout(PredictionForm)
        self.verticalLayout.setObjectName("verticalLayout")
        self.textBrowser = QtWidgets.QTextBrowser(PredictionForm)
        self.textBrowser.setMinimumSize(QtCore.QSize(601, 110))
        self.textBrowser.setMaximumSize(QtCore.QSize(16777215, 110))
        self.textBrowser.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout.addWidget(self.textBrowser)
        self.tableWidget = QtWidgets.QTableWidget(PredictionForm)
        self.tableWidget.setMinimumSize(QtCore.QSize(0, 65))
        self.tableWidget.setMaximumSize(QtCore.QSize(16777215, 65))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(1)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        self.verticalLayout.addWidget(self.tableWidget)
        self.tableView = QtWidgets.QTableView(PredictionForm)
        self.tableView.setObjectName("tableView")
        self.verticalLayout.addWidget(self.tableView)
        self.textResults = QtWidgets.QTextBrowser(PredictionForm)
        self.textResults.setObjectName("textResults")
        self.verticalLayout.addWidget(self.textResults)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(PredictionForm)
        self.label_2.setMinimumSize(QtCore.QSize(143, 16))
        self.label_2.setMaximumSize(QtCore.QSize(143, 16))
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.spinConfidence = QtWidgets.QSpinBox(PredictionForm)
        self.spinConfidence.setMinimumSize(QtCore.QSize(45, 23))
        self.spinConfidence.setMaximumSize(QtCore.QSize(45, 23))
        self.spinConfidence.setMinimum(75)
        self.spinConfidence.setMaximum(99)
        self.spinConfidence.setProperty("value", 95)
        self.spinConfidence.setObjectName("spinConfidence")
        self.horizontalLayout_2.addWidget(self.spinConfidence)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.checkBox = QtWidgets.QCheckBox(PredictionForm)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout_2.addWidget(self.checkBox)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(PredictionForm)
        self.label.setMinimumSize(QtCore.QSize(91, 16))
        self.label.setMaximumSize(QtCore.QSize(91, 16))
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.spinBox = QtWidgets.QSpinBox(PredictionForm)
        self.spinBox.setMinimumSize(QtCore.QSize(90, 23))
        self.spinBox.setMinimum(10)
        self.spinBox.setMaximum(10000)
        self.spinBox.setProperty("value", 100)
        self.spinBox.setObjectName("spinBox")
        self.horizontalLayout.addWidget(self.spinBox)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.btnCreate = QtWidgets.QPushButton(PredictionForm)
        self.btnCreate.setMinimumSize(QtCore.QSize(75, 23))
        self.btnCreate.setMaximumSize(QtCore.QSize(75, 23))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/btn_ok_200.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnCreate.setIcon(icon1)
        self.btnCreate.setObjectName("btnCreate")
        self.horizontalLayout.addWidget(self.btnCreate)
        self.btnSave = QtWidgets.QPushButton(PredictionForm)
        self.btnSave.setEnabled(False)
        self.btnSave.setMinimumSize(QtCore.QSize(91, 23))
        self.btnSave.setMaximumSize(QtCore.QSize(91, 23))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnSave.setIcon(icon2)
        self.btnSave.setObjectName("btnSave")
        self.horizontalLayout.addWidget(self.btnSave)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(PredictionForm)
        QtCore.QMetaObject.connectSlotsByName(PredictionForm)

    def retranslateUi(self, PredictionForm):
        _translate = QtCore.QCoreApplication.translate
        PredictionForm.setWindowTitle(_translate("PredictionForm", "Прогнозирование по модели"))
        self.textBrowser.setHtml(_translate("PredictionForm", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Franklin Gothic Medium\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/icons/help.png\" width=\"15\" height=\"16\" style=\"vertical-align: top;\" /><span style=\" vertical-align:top;\">Прогнозирование по модели используется для определения состояния объекта при определенных входных управляемых воздействиях. Метод использует линейную регрессионную модель. В качестве результата выводит оценки разброса в виде доверительного интервала и оценки центрального значения (среднее для количественных и медиана для порядковых).</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Позволяет работать <span style=\" text-decoration: underline;\">с количественными и порядковыми факторами</span>.</p></body></html>"))
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("PredictionForm", "Начальное состояние"))
        self.label_2.setText(_translate("PredictionForm", "Доверительный уровень"))
        self.checkBox.setText(_translate("PredictionForm", "Фиксировать наблюдаемые факторы"))
        self.label.setText(_translate("PredictionForm", "Объем данных"))
        self.btnCreate.setText(_translate("PredictionForm", "Создать"))
        self.btnSave.setText(_translate("PredictionForm", "Сохранить"))
import forms_resources_rc
