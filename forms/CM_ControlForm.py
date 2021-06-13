# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CM_ControlForm.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ControlForm(object):
    def setupUi(self, ControlForm):
        ControlForm.setObjectName("ControlForm")
        ControlForm.resize(620, 620)
        ControlForm.setMinimumSize(QtCore.QSize(620, 620))
        font = QtGui.QFont()
        font.setFamily("Franklin Gothic Medium")
        font.setPointSize(10)
        ControlForm.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/control.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        ControlForm.setWindowIcon(icon)
        self.verticalLayout = QtWidgets.QVBoxLayout(ControlForm)
        self.verticalLayout.setObjectName("verticalLayout")
        self.textBrowser = QtWidgets.QTextBrowser(ControlForm)
        self.textBrowser.setMinimumSize(QtCore.QSize(601, 110))
        self.textBrowser.setMaximumSize(QtCore.QSize(16777215, 110))
        self.textBrowser.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout.addWidget(self.textBrowser)
        self.tableWidget = QtWidgets.QTableWidget(ControlForm)
        self.tableWidget.setMinimumSize(QtCore.QSize(0, 65))
        self.tableWidget.setMaximumSize(QtCore.QSize(16777215, 65))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(1)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        self.verticalLayout.addWidget(self.tableWidget)
        self.tableTargeted = QtWidgets.QTableWidget(ControlForm)
        self.tableTargeted.setMinimumSize(QtCore.QSize(0, 105))
        self.tableTargeted.setMaximumSize(QtCore.QSize(16777215, 105))
        self.tableTargeted.setObjectName("tableTargeted")
        self.tableTargeted.setColumnCount(0)
        self.tableTargeted.setRowCount(2)
        item = QtWidgets.QTableWidgetItem()
        self.tableTargeted.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableTargeted.setVerticalHeaderItem(1, item)
        self.verticalLayout.addWidget(self.tableTargeted)
        self.tableView = QtWidgets.QTableView(ControlForm)
        self.tableView.setObjectName("tableView")
        self.verticalLayout.addWidget(self.tableView)
        self.textResults = QtWidgets.QTextBrowser(ControlForm)
        self.textResults.setObjectName("textResults")
        self.verticalLayout.addWidget(self.textResults)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(ControlForm)
        self.label.setMinimumSize(QtCore.QSize(91, 16))
        self.label.setMaximumSize(QtCore.QSize(91, 16))
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.spinBox = QtWidgets.QSpinBox(ControlForm)
        self.spinBox.setMinimumSize(QtCore.QSize(90, 23))
        self.spinBox.setMinimum(10)
        self.spinBox.setMaximum(10000)
        self.spinBox.setProperty("value", 100)
        self.spinBox.setObjectName("spinBox")
        self.horizontalLayout.addWidget(self.spinBox)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.btnCreate = QtWidgets.QPushButton(ControlForm)
        self.btnCreate.setMinimumSize(QtCore.QSize(75, 23))
        self.btnCreate.setMaximumSize(QtCore.QSize(75, 23))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/btn_ok_200.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnCreate.setIcon(icon1)
        self.btnCreate.setObjectName("btnCreate")
        self.horizontalLayout.addWidget(self.btnCreate)
        self.btnSave = QtWidgets.QPushButton(ControlForm)
        self.btnSave.setEnabled(False)
        self.btnSave.setMinimumSize(QtCore.QSize(91, 23))
        self.btnSave.setMaximumSize(QtCore.QSize(91, 23))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnSave.setIcon(icon2)
        self.btnSave.setObjectName("btnSave")
        self.horizontalLayout.addWidget(self.btnSave)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(ControlForm)
        QtCore.QMetaObject.connectSlotsByName(ControlForm)

    def retranslateUi(self, ControlForm):
        _translate = QtCore.QCoreApplication.translate
        ControlForm.setWindowTitle(_translate("ControlForm", "Определение управления"))
        self.textBrowser.setHtml(_translate("ControlForm", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Franklin Gothic Medium\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/icons/help.png\" width=\"15\" height=\"16\" style=\"vertical-align: top;\" /><span style=\" vertical-align:top;\">Определение управления применяется для нахождения значений управляемых факторов, позволяющих добиться нужных значений целевых факторов. Для применения требуется начальное состояние объекта, а также желаемые значения целевых факторов и их веса. Веса определяют приоритет целевых факторов. В качестве метода поиска применяется градиентный спуск.</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Позволяет работать <span style=\" text-decoration: underline;\">с количественными и порядковыми факторами</span>.</p></body></html>"))
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("ControlForm", "Начальное состояние"))
        item = self.tableTargeted.verticalHeaderItem(0)
        item.setText(_translate("ControlForm", "Желаемые значения"))
        item = self.tableTargeted.verticalHeaderItem(1)
        item.setText(_translate("ControlForm", "Веса"))
        self.label.setText(_translate("ControlForm", "Объем данных"))
        self.btnCreate.setText(_translate("ControlForm", "Создать"))
        self.btnSave.setText(_translate("ControlForm", "Сохранить"))
import forms_resources_rc