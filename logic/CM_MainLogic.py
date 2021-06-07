# -*- coding: utf-8 -*-

from os import path
import json

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

from forms.CM_MainForm import Ui_MainWindow
from model.CM_classes import CognitiveModel, OrdinalFactor, NominalFactor, QuantitativeFactor
from logic.CM_EditLogic import EditWindow
from logic.CM_TableLogic import TableWindow
from logic.CM_GraphLogic import GraphWindow


class Test_Thread(QtCore.QThread):
    testSignal = QtCore.pyqtSignal()

    def __init__(self, tab):
        QtCore.QThread.__init__(self)
        self.tab = tab

    def run(self):
        while True:
            self.msleep(2000)
            # self.testSignal.emit()


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        # Setup
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.cognitive_model = CognitiveModel('Новая модель', '')
        self.file = ""
        self.saved = True

        # Play with colors
        self.palette_origin = self.palette()
        self.palette_dark = QPalette()
        self.palette_dark.setColor(QPalette.Window, QColor(53, 53, 53))
        self.palette_dark.setColor(QPalette.WindowText, Qt.white)
        self.palette_dark.setColor(QPalette.Base, QColor(25, 25, 25))
        self.palette_dark.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        self.palette_dark.setColor(QPalette.ToolTipBase, Qt.white)
        self.palette_dark.setColor(QPalette.ToolTipText, Qt.white)
        self.palette_dark.setColor(QPalette.Text, Qt.white)
        self.palette_dark.setColor(QPalette.Button, QColor(53, 53, 53))
        self.palette_dark.setColor(QPalette.ButtonText, Qt.white)
        self.palette_dark.setColor(QPalette.BrightText, QColor(53, 255, 53))
        self.palette_dark.setColor(QPalette.Link, QColor(42, 130, 218))
        self.palette_dark.setColor(QPalette.Highlight, QColor(100, 0, 0))
        self.palette_dark.setColor(QPalette.HighlightedText, Qt.white)
        self.is_origin_palette = True

        # Debug
        self.test_thread = Test_Thread(None)
        self.test_thread.testSignal.connect(sum)
        self.test_thread.start()

        # Connecting slots to action signals
        self.action_edit.triggered.connect(self.create_edit_window)
        self.action_clear_links.triggered.connect(self.clear_links)
        self.action_clear_factors.triggered.connect(self.clear_factors)
        self.action_table.triggered.connect(self.create_table_window)
        self.action_graph.triggered.connect(self.create_graph_window)
        self.action_save_as.triggered.connect(self.model_save_as)
        self.action_save.triggered.connect(self.model_save)
        self.action_open.triggered.connect(self.model_open)
        self.action_Qt.triggered.connect(self.about_Qt)
        self.action_exit.triggered.connect(self.exit)
        self.action_new.triggered.connect(self.new)

        self.action_theme.triggered.connect(self.change_theme)

        # test model
        self.cognitive_model.name = "Тестовая модель корпорации"
        self.cognitive_model.description = "Здесь дано тестовое описание модели.\n" \
                                           "Модель состоит из четырех факторов: " \
                                           "по одному каждой роли, два количественных, " \
                                           "один номинальный и один порядковый.\n" \
                                           "Никакого смысла в модели нет."
        factor1 = QuantitativeFactor('Финансирование', 0, max_value=200000, min_value=5000)
        self.cognitive_model.add_factor(factor1)
        factor2 = QuantitativeFactor('Потери бюджета', 1, max_value=100000, min_value=1000)
        self.cognitive_model.add_factor(factor2)
        factor3 = OrdinalFactor('Качество', 2, max_value=7, ranks_probabilities=[0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1])
        self.cognitive_model.add_factor(factor3)
        factor4 = NominalFactor('Состояние', 3, cats=['Изменилось', 'Не изменилось', 'Неизвестно'],
                                cats_probabilities=[0.2, 0.5, 0.3])
        self.cognitive_model.add_factor(factor4)
        self.cognitive_model.add_link(factor1, factor3, 0.9)
        self.cognitive_model.add_link(factor2, factor3, -0.4)
        self.cognitive_model.add_link(factor3, factor4, 0.75)
        self.cognitive_model.add_link(factor4, factor3, 0.5)

    def update_children(self):
        if not self.file:
            title = "Когнитивное моделирование систем управления"
        else:
            title = "%s - Когнитивное моделирование систем управления" % path.basename(self.file)
        self.setWindowTitle(title)
        self.saved = False
        for window in self.mdiArea.subWindowList():
            window.widget().model_changed()

    def create_window(self, WindowClass, icon_path=None):
        if len(self.mdiArea.subWindowList()) > 4:
            QtWidgets.QMessageBox.information(self,
                                              "Слишком много окон",
                                              "Разрешено не более 5 окон!",
                                              buttons=QtWidgets.QMessageBox.Ok)
            return
        for window in self.mdiArea.subWindowList():
            if isinstance(window.widget(), WindowClass):
                window.show()
                self.mdiArea.setActiveSubWindow(window)
                window.move(0, 0)
                return
        w = WindowClass(self.cognitive_model)
        w.modelChangedSignal.connect(self.update_children)
        sub = self.mdiArea.addSubWindow(w)
        sub.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        sub.setWindowIcon(QtGui.QIcon(icon_path))
        sub.move(0, 0)
        sub.show()

    def create_edit_window(self):
        try:
            self.create_window(EditWindow, ":/icons/edit.png")
        except Exception as E:
            print(E)

    def create_table_window(self):
        self.create_window(TableWindow, ":/icons/table.png")

    def create_graph_window(self):
        self.create_window(GraphWindow, ":/icons/graph.png")

    def new(self):
        buttons = QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel
        if not self.saved:
            result = QtWidgets.QMessageBox.question(self, "Данные не сохранены!",
                                                    "Сохранить модель перед созданием новой?",
                                                    buttons=buttons,
                                                    defaultButton=QtWidgets.QMessageBox.Cancel)
            if result == QtWidgets.QMessageBox.Yes:
                if not self.model_save():
                    return
            elif result == QtWidgets.QMessageBox.No:
                pass
            elif result == QtWidgets.QMessageBox.Cancel:
                return
        self.cognitive_model.clear_factors()
        self.cognitive_model.name = "Новая модель"
        self.cognitive_model.description = ""
        self.file = None
        self.update_children()

    def clear_links(self):
        buttons = QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel
        result = QtWidgets.QMessageBox.question(self, "Вопрос",
                                                "Удалить все связи из модели?",
                                                buttons=buttons,
                                                defaultButton=QtWidgets.QMessageBox.Cancel)
        if result == QtWidgets.QMessageBox.Yes:
            self.cognitive_model.clear_links()
            self.update_children()

    def clear_factors(self):
        result = QtWidgets.QMessageBox.question(self, "Вопрос",
                                                "Удалить все факторы из модели?\n"
                                                "Это также удалит все связи.",
                                                buttons=QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
                                                defaultButton=QtWidgets.QMessageBox.Cancel)
        if result == QtWidgets.QMessageBox.Yes:
            self.cognitive_model.clear_factors()
            self.update_children()

    def try_save(self, file):
        if file:
            cm_json = self.cognitive_model.encode_json()
            try:
                # json.dump(cm_json, open(file, 'w'), indent=4, ensure_ascii=False)
                json.dump(cm_json, open(file, 'w'), indent=4, ensure_ascii=True)
            except OSError:
                QtWidgets.QMessageBox.critical(self,
                                               "Ошибка сохранения",
                                               "Не удалось сохранить модель в указанный файл: %s" % file,
                                               buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.file = file
                self.update_children()
                self.saved = True
                return True

    def model_save_as(self):
        filename, filetype = QtWidgets.QFileDialog.getSaveFileName(parent=self,
                                                                   filter="JavaScript Object Notation (*.json);;"
                                                                          "All (*);;Text files (*.txt)")
        if filetype == "JavaScript Object Notation (*.json)" and filename[-5:] != '.json':
            filename = filename + '.json'
        elif filetype == "Text files (*.txt)" and filename[-4:] != '.txt':
            filename = filename + '.txt'
        return self.try_save(filename)

    def model_save(self):
        if self.file:
            return self.try_save(self.file)
        else:
            return self.model_save_as()

    def model_open(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(parent=self,
                                                             filter="JavaScript Object Notation (*.json);;"
                                                                    "All (*);;Text files (*.txt)")
        if file_name:
            try:
                json_cm = json.load(open(file_name, 'r'))
                self.cognitive_model.load_json(json_cm)
            except OSError:
                QtWidgets.QMessageBox.critical(self,
                                               "Ошибка открытия",
                                               "Не удалось открыть указанный файл: %s" % file_name,
                                               buttons=QtWidgets.QMessageBox.Ok)
            except ValueError:
                QtWidgets.QMessageBox.critical(self,
                                               "Ошибка загрузки",
                                               "Не удалось загрузить модель из указанного файла: %s" % file_name,
                                               buttons=QtWidgets.QMessageBox.Ok)
            except Exception as E:
                print(E)
            else:
                self.file = file_name
                self.update_children()
                self.saved = True

    def about_Qt(self):
        QtWidgets.QMessageBox.aboutQt(self)

    def exit(self):
        if not self.saved:
            buttons = QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel
            result = QtWidgets.QMessageBox.question(self, "Данные не сохранены",
                                                    "Сохранить модель?",
                                                    buttons=buttons,
                                                    defaultButton=QtWidgets.QMessageBox.Cancel)
            if result == QtWidgets.QMessageBox.Yes:
                if not self.model_save():
                    return
            elif result == QtWidgets.QMessageBox.No:
                pass
            elif result == QtWidgets.QMessageBox.Cancel:
                return
        QtWidgets.qApp.quit()

    def change_theme(self):
        if self.is_origin_palette:
            self.setPalette(self.palette_dark)
        else:
            self.setPalette(self.palette_origin)
        self.is_origin_palette = not self.is_origin_palette
