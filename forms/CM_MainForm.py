# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CM_MainForm.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setMinimumSize(QtCore.QSize(800, 600))
        font = QtGui.QFont()
        font.setFamily("Franklin Gothic Medium")
        font.setPointSize(10)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/main.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.mdiArea = QtWidgets.QMdiArea(self.centralwidget)
        self.mdiArea.setFrameShape(QtWidgets.QFrame.Box)
        self.mdiArea.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.mdiArea.setObjectName("mdiArea")
        self.gridLayout.addWidget(self.mdiArea, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menubar)
        self.menu_4.setObjectName("menu_4")
        self.menu_5 = QtWidgets.QMenu(self.menubar)
        self.menu_5.setObjectName("menu_5")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.action_new = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/new.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_new.setIcon(icon1)
        self.action_new.setObjectName("action_new")
        self.action_open = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_open.setIcon(icon2)
        self.action_open.setObjectName("action_open")
        self.action_save = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_save.setIcon(icon3)
        self.action_save.setObjectName("action_save")
        self.action_save_as = QtWidgets.QAction(MainWindow)
        self.action_save_as.setObjectName("action_save_as")
        self.action_exit = QtWidgets.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/exit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_exit.setIcon(icon4)
        self.action_exit.setObjectName("action_exit")
        self.action_clear_links = QtWidgets.QAction(MainWindow)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/icons/clear-links.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_clear_links.setIcon(icon5)
        self.action_clear_links.setObjectName("action_clear_links")
        self.action_clear_factors = QtWidgets.QAction(MainWindow)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/icons/clear-factors.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_clear_factors.setIcon(icon6)
        self.action_clear_factors.setObjectName("action_clear_factors")
        self.action_edit = QtWidgets.QAction(MainWindow)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/icons/edit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_edit.setIcon(icon7)
        self.action_edit.setObjectName("action_edit")
        self.action_imitation = QtWidgets.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/icons/imitation.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_imitation.setIcon(icon8)
        self.action_imitation.setObjectName("action_imitation")
        self.action_prediction = QtWidgets.QAction(MainWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/icons/prediction.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_prediction.setIcon(icon9)
        self.action_prediction.setObjectName("action_prediction")
        self.action_control = QtWidgets.QAction(MainWindow)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(":/icons/control.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_control.setIcon(icon10)
        self.action_control.setObjectName("action_control")
        self.action_table = QtWidgets.QAction(MainWindow)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap(":/icons/table.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_table.setIcon(icon11)
        self.action_table.setObjectName("action_table")
        self.action_graph = QtWidgets.QAction(MainWindow)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap(":/icons/graph.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_graph.setIcon(icon12)
        self.action_graph.setObjectName("action_graph")
        self.action_process = QtWidgets.QAction(MainWindow)
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap(":/icons/process.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_process.setIcon(icon13)
        self.action_process.setObjectName("action_process")
        self.action_help = QtWidgets.QAction(MainWindow)
        icon14 = QtGui.QIcon()
        icon14.addPixmap(QtGui.QPixmap(":/icons/help.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_help.setIcon(icon14)
        self.action_help.setObjectName("action_help")
        self.action_about = QtWidgets.QAction(MainWindow)
        self.action_about.setObjectName("action_about")
        self.action_Qt = QtWidgets.QAction(MainWindow)
        icon15 = QtGui.QIcon()
        icon15.addPixmap(QtGui.QPixmap(":/icons/Qt.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_Qt.setIcon(icon15)
        self.action_Qt.setObjectName("action_Qt")
        self.action_theme = QtWidgets.QAction(MainWindow)
        icon16 = QtGui.QIcon()
        icon16.addPixmap(QtGui.QPixmap(":/icons/Theme.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_theme.setIcon(icon16)
        self.action_theme.setObjectName("action_theme")
        self.menu.addAction(self.action_new)
        self.menu.addSeparator()
        self.menu.addAction(self.action_open)
        self.menu.addSeparator()
        self.menu.addAction(self.action_save)
        self.menu.addAction(self.action_save_as)
        self.menu.addSeparator()
        self.menu.addAction(self.action_exit)
        self.menu_2.addAction(self.action_edit)
        self.menu_2.addAction(self.action_clear_links)
        self.menu_2.addAction(self.action_clear_factors)
        self.menu_3.addAction(self.action_imitation)
        self.menu_3.addSeparator()
        self.menu_3.addAction(self.action_process)
        self.menu_3.addSeparator()
        self.menu_3.addAction(self.action_prediction)
        self.menu_3.addAction(self.action_control)
        self.menu_4.addAction(self.action_table)
        self.menu_4.addAction(self.action_graph)
        self.menu_5.addAction(self.action_help)
        self.menu_5.addSeparator()
        self.menu_5.addAction(self.action_about)
        self.menu_5.addAction(self.action_Qt)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_4.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_5.menuAction())
        self.toolBar.addAction(self.action_new)
        self.toolBar.addAction(self.action_open)
        self.toolBar.addAction(self.action_save)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.action_edit)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.action_table)
        self.toolBar.addAction(self.action_graph)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.action_imitation)
        self.toolBar.addAction(self.action_process)
        self.toolBar.addAction(self.action_prediction)
        self.toolBar.addAction(self.action_control)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.action_theme)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Когнитивное моделирование систем управления"))
        self.menu.setTitle(_translate("MainWindow", "Файл"))
        self.menu_2.setTitle(_translate("MainWindow", "Изменение"))
        self.menu_3.setTitle(_translate("MainWindow", "Применение"))
        self.menu_4.setTitle(_translate("MainWindow", "Визуализация"))
        self.menu_5.setTitle(_translate("MainWindow", "Справка"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.action_new.setText(_translate("MainWindow", "Новая модель..."))
        self.action_new.setToolTip(_translate("MainWindow", "Создать пустую модель"))
        self.action_new.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.action_open.setText(_translate("MainWindow", "Открыть..."))
        self.action_open.setToolTip(_translate("MainWindow", "Открыть файл с моделью"))
        self.action_open.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.action_save.setText(_translate("MainWindow", "Сохранить"))
        self.action_save.setToolTip(_translate("MainWindow", "Сохранить текущую модель в файл"))
        self.action_save.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.action_save_as.setText(_translate("MainWindow", "Сохранить как..."))
        self.action_save_as.setToolTip(_translate("MainWindow", "Выбрать файл для сохранения модели"))
        self.action_save_as.setShortcut(_translate("MainWindow", "Ctrl+Shift+S"))
        self.action_exit.setText(_translate("MainWindow", "Выход"))
        self.action_exit.setToolTip(_translate("MainWindow", "Завершить работу с приложением"))
        self.action_exit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.action_clear_links.setText(_translate("MainWindow", "Очистить связи"))
        self.action_clear_links.setToolTip(_translate("MainWindow", "Удалить все связи из модели"))
        self.action_clear_links.setShortcut(_translate("MainWindow", "Ctrl+X"))
        self.action_clear_factors.setText(_translate("MainWindow", "Очистить факторы"))
        self.action_clear_factors.setShortcut(_translate("MainWindow", "Ctrl+Shift+X"))
        self.action_edit.setText(_translate("MainWindow", "Редактирование модели"))
        self.action_edit.setToolTip(_translate("MainWindow", "Приступить к изменению описания модели, её факторов и связей"))
        self.action_edit.setShortcut(_translate("MainWindow", "Ctrl+E"))
        self.action_imitation.setText(_translate("MainWindow", "Создание имитационных данных"))
        self.action_imitation.setToolTip(_translate("MainWindow", "Начать процесс формирования имитационных данных"))
        self.action_imitation.setShortcut(_translate("MainWindow", "Ctrl+I"))
        self.action_prediction.setText(_translate("MainWindow", "Прогнозирование по модели"))
        self.action_prediction.setToolTip(_translate("MainWindow", "Определение последствий подачи управляющих воздействий"))
        self.action_prediction.setShortcut(_translate("MainWindow", "Ctrl+P"))
        self.action_control.setText(_translate("MainWindow", "Определение управления"))
        self.action_control.setToolTip(_translate("MainWindow", "Определение нужных управляющих воздействий для приближения к заданным значениям целевых факторов"))
        self.action_control.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.action_table.setText(_translate("MainWindow", "Табличный вид"))
        self.action_table.setToolTip(_translate("MainWindow", "Окно с табличным представлением когнитивной модели"))
        self.action_table.setShortcut(_translate("MainWindow", "Ctrl+T"))
        self.action_graph.setText(_translate("MainWindow", "Граф"))
        self.action_graph.setToolTip(_translate("MainWindow", "Графовое представление когнитивной модели"))
        self.action_graph.setShortcut(_translate("MainWindow", "Ctrl+G"))
        self.action_process.setText(_translate("MainWindow", "Оценка переходного процесса"))
        self.action_process.setToolTip(_translate("MainWindow", "Оценка переходного процесса в системе, состоящей только из количественных факторов"))
        self.action_process.setShortcut(_translate("MainWindow", "Ctrl+R"))
        self.action_help.setText(_translate("MainWindow", "Помощь"))
        self.action_help.setToolTip(_translate("MainWindow", "Подробное руководство для использования программы"))
        self.action_help.setShortcut(_translate("MainWindow", "F1"))
        self.action_about.setText(_translate("MainWindow", "О программе"))
        self.action_about.setToolTip(_translate("MainWindow", "Сведения о программе, ей версии и авторе"))
        self.action_Qt.setText(_translate("MainWindow", "О Qt"))
        self.action_theme.setText(_translate("MainWindow", "Сменить тему"))
        self.action_theme.setToolTip(_translate("MainWindow", "Меняет тему приложения"))
import forms_resources_rc
