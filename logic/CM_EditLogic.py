# -*- coding: utf-8 -*-

from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import Qt

from forms.CM_EditForm import Ui_EditForm
from logic.CM_Abstracts import AbstractLogic
from model.CM_classes import InterfactorLink, AbstractFactor, QuantitativeFactor, OrdinalFactor, NominalFactor
from model.CM_funcs import number_2_quality, quality_2_number


class EditWindow(AbstractLogic, Ui_EditForm):

    def __init__(self, cognitive_model):
        super(EditWindow, self).__init__(cognitive_model)
        self.setupUi(self)
        self.model_changed()

        # Set form params
        self.spinRanks.setMaximum(20)
        self.spinNominals.setMaximum(20)
        self.spinNominals.setValue(3)
        self.spinRanks.setValue(3)
        self.ranks_changed(3)
        self.cats_changed(3)
        self.current_link = None
        self.current_factor = None

        # Set text parameters
        self.textEdit.setPlaceholderText('Не более 300 символов')
        self.nameEdit.setPlaceholderText('Не более 50 символов')
        self.nameEditFactor.setPlaceholderText('Не более 50 символов')
        reg = QtCore.QRegExp("[а-яА-Я0-9a-zA-ZёЁ \\/\\.\\,\\*\\(\\)\\[\\]\\{\\}\\$\\\\\\%\\№\\#\\!\\?\\-\\+]{50}")
        validator = QtGui.QRegExpValidator(reg)
        self.nameEdit.setValidator(validator)
        self.nameEditFactor.setValidator(validator)

        # Connect events on signals for description group
        self.btnSort.clicked.connect(self.clicked_sort)
        self.btnChangeText.clicked.connect(self.clicked_change_text)
        # Connect events on signals for factors group
        self.listFactors.itemClicked.connect(self.factor_selected)
        self.spinRanks.valueChanged.connect(self.ranks_changed)
        self.spinNominals.valueChanged.connect(self.cats_changed)
        self.btnDeleteFactor.clicked.connect(self.clicked_delete_factor)
        self.btnAddFactor.clicked.connect(self.clicked_add_factor)
        # Connect events on signals for links group
        self.comboFactorFrom.currentIndexChanged.connect(self.check_factors_scales)
        self.comboFactorTo.currentIndexChanged.connect(self.check_factors_scales)
        self.listLinks.itemClicked.connect(self.link_selected)
        self.spinStrength.valueChanged.connect(self.check_quality)
        self.comboQuality.activated.connect(self.check_strength)
        self.btnDeleteLink.clicked.connect(self.clicked_delete_link)
        self.btnAcceptLink.clicked.connect(self.clicked_accept_link)

    def model_changed(self):
        # Change form
        self.comboFactorTo.clear()
        self.comboFactorFrom.clear()
        self.listLinks.clear()
        self.listFactors.clear()
        self.btnDeleteLink.setDisabled(True)
        self.btnDeleteFactor.setDisabled(True)
        # Text
        self.nameEdit.setText(self.cognitive_model.name)
        self.textEdit.setPlainText(self.cognitive_model.description)
        # Change factors
        factors: list[AbstractFactor] = self.cognitive_model.get_factors()
        for f in factors:
            item = QtWidgets.QListWidgetItem(str(f))
            item.setData(Qt.UserRole + 1, f)
            self.listFactors.addItem(item)
            self.comboFactorTo.addItem(str(f.get_id()), f)
            self.comboFactorFrom.addItem(str(f.get_id()), f)
        # Change links
        links = self.cognitive_model.get_links()
        for L in links:
            item = QtWidgets.QListWidgetItem(str(L))
            item.setData(Qt.UserRole + 1, L)
            self.listLinks.addItem(item)

    def check_factors_scales(self, ind):
        i = self.comboFactorFrom.currentIndex()
        j = self.comboFactorTo.currentIndex()
        if i < 0 or j < 0:
            pass
        factor_from = self.comboFactorFrom.itemData(i)
        factor_to = self.comboFactorTo.itemData(j)
        if factor_from and factor_to:
            self.check_nominals_link(factor_from, factor_to)

    def link_selected(self, item: QtWidgets.QListWidgetItem):
        link: InterfactorLink = item.data(Qt.UserRole + 1)
        self.current_link = link
        self.btnDeleteLink.setEnabled(True)
        self.spinStrength.setValue(abs(round(link.strength * 100)))
        if link.strength > 0:
            self.radioPositive.setChecked(True)
        else:
            self.radioNegative.setChecked(True)
        self.check_nominals_link(link.factor_from, link.factor_to)
        i, j = link.get_factors_numbers()
        self.comboFactorFrom.setCurrentIndex(i - 1)
        self.comboFactorTo.setCurrentIndex(j - 1)

    def check_nominals_link(self, factor_from, factor_to):
        scale = max(factor_to.scale, factor_from.scale)
        if scale == 2:
            self.radioPositive.setChecked(True)
            self.radioPositive.setDisabled(True)
            self.radioNegative.setDisabled(True)
        else:
            self.radioPositive.setEnabled(True)
            self.radioNegative.setEnabled(True)

    def check_quality(self, strength: int):
        self.comboQuality.setCurrentIndex(number_2_quality(strength))

    def check_strength(self, index: int):
        self.spinStrength.setValue(quality_2_number(index))

    def clicked_delete_link(self):
        if not self.current_link:
            self.btnDeleteLink.setDisabled(True)
        self.cognitive_model.del_link(self.current_link)
        self.notify_main()

    def clicked_accept_link(self):
        i = self.comboFactorFrom.currentIndex()
        j = self.comboFactorTo.currentIndex()
        if i < 0 or j < 0:
            pass
        if i == j:
            QtWidgets.QMessageBox.information(self,
                                              "Неправильная связь",
                                              "Петли запрещены",
                                              buttons=QtWidgets.QMessageBox.Ok)
            return
        factor_from = self.comboFactorFrom.itemData(i)
        factor_to = self.comboFactorTo.itemData(j)
        if factor_to.role < 2:
            QtWidgets.QMessageBox.information(self,
                                              "Неправильная связь",
                                              "Входной фактор не может быть зависимым",
                                              buttons=QtWidgets.QMessageBox.Ok)
            return
        strength = self.spinStrength.value()
        sign = 1 if self.radioPositive.isChecked() else -1
        self.cognitive_model.add_link(factor_from, factor_to, strength * sign / 100)
        self.notify_main()

    def clicked_sort(self):
        if self.cognitive_model.check_cycles():
            QtWidgets.QMessageBox.information(self,
                                              "Обнаружен цикл",
                                              "В модели обнаружены циклы, поэтому топологическая "
                                              "сортировка факторов невозможна!",
                                              buttons=QtWidgets.QMessageBox.Ok)
            return
        self.cognitive_model.sort_factors()
        self.notify_main()

    def clicked_change_text(self):
        name = self.nameEdit.text()
        if not name:
            QtWidgets.QMessageBox.information(self,
                                              "Ошибка ввода",
                                              "Отсутствует имя модели",
                                              buttons=QtWidgets.QMessageBox.Ok)
            return
        self.cognitive_model.name = name
        self.cognitive_model.description = self.textEdit.document().toPlainText()[:300]

    def factor_selected(self, item: QtWidgets.QListWidgetItem):
        factor: AbstractFactor = item.data(Qt.UserRole + 1)
        self.current_factor = factor
        self.btnDeleteFactor.setEnabled(True)
        self.comboRole.setCurrentIndex(factor.role)
        self.comboScale.setCurrentIndex(factor.scale)
        self.nameEditFactor.setText(factor.name)
        if factor.scale == 0:
            factor: QuantitativeFactor
            self.spinMax.setValue(factor.max_value)
            self.spinMin.setValue(factor.min_value)
            self.spinMean.setValue(factor.mean)
            self.spinDeviation.setValue(factor.standard_deviation)
        elif factor.scale == 1:
            factor: OrdinalFactor
            self.spinRanks.setValue(factor.max_value)
            self.ranks_changed(factor.max_value)
            for i, prob in enumerate(factor.ranks_probabilities):
                self.tableRanks.cellWidget(i, 0).setValue(round(prob * 100))
        elif factor.scale == 2:
            factor: NominalFactor
            self.spinNominals.setValue(len(factor.cats))
            self.cats_changed(len(factor.cats))
            for i, prob in enumerate(factor.cats_probabilities):
                self.tableNominals.cellWidget(i, 1).setValue(round(prob * 100))
                self.tableNominals.cellWidget(i, 0).setText(factor.cats[i])

    def ranks_changed(self, max_rank: int):
        n = self.tableRanks.rowCount()
        if self.tableRanks.rowCount() == max_rank:
            return
        self.tableRanks.setRowCount(max_rank)
        for i in range(n, max_rank):
            spin = QtWidgets.QSpinBox()
            spin.setMaximum(99)
            spin.setMinimum(1)
            spin.setValue(100//max_rank)
            self.tableRanks.setCellWidget(i, 0, spin)

    def cats_changed(self, cats_count: int):
        n = self.tableNominals.rowCount()
        if self.tableNominals.rowCount() == cats_count:
            return
        self.tableNominals.setRowCount(cats_count)
        for i in range(n, cats_count):
            # Prob of nominal
            spin = QtWidgets.QSpinBox()
            spin.setMaximum(99)
            spin.setMinimum(1)
            spin.setValue(100//cats_count)
            self.tableNominals.setCellWidget(i, 1, spin)
            # Name of nominal
            line = QtWidgets.QLineEdit()
            reg = QtCore.QRegExp("[а-яА-Я0-9a-zA-ZёЁ \\/\\.\\,\\*\\(\\)\\[\\]\\{\\}\\$\\\\\\%\\№\\#\\!\\?\\-\\+]{50}")
            validator = QtGui.QRegExpValidator(reg)
            line.setValidator(validator)
            line.setText(chr(ord('A') + i))
            self.tableNominals.setCellWidget(i, 0, line)

    def clicked_delete_factor(self):
        if not self.current_factor:
            self.btnDeleteFactor.setDisabled(True)
        self.cognitive_model.del_factor(self.current_factor)
        self.notify_main()

    def clicked_add_factor(self):
        name = self.nameEditFactor.text()
        if not name:
            QtWidgets.QMessageBox.information(self,
                                              "Ошибка ввода",
                                              "Отсутствует имя фактора",
                                              buttons=QtWidgets.QMessageBox.Ok)
            return
        role = self.comboRole.currentIndex()
        scale = self.comboScale.currentIndex()
        if role < 0 or role < 0:
            return
        if scale == 0:
            min_value = self.spinMin.value()
            max_value = self.spinMax.value()
            mean = self.spinMean.value()
            deviation = self.spinDeviation.value()
            factor = QuantitativeFactor(name, role, max_value=max_value, min_value=min_value, mean=mean,
                                        standard_deviation=deviation)
        elif scale == 1:
            max_value = self.spinRanks.value()
            ranks_probabilities = [self.tableRanks.cellWidget(i, 0).value()/100 for i in range(max_value)]
            factor = OrdinalFactor(name, role, max_value=max_value, ranks_probabilities=ranks_probabilities)
        else:
            count = self.spinNominals.value()
            cats = []
            cats_probabilities = []
            for i in range(count):
                text = self.tableNominals.cellWidget(i, 0).text().strip()
                prob = self.tableNominals.cellWidget(i, 1).value()/100
                if not text:
                    QtWidgets.QMessageBox.information(self,
                                                      "Ошибка ввода",
                                                      "Отсутствует номинал",
                                                      buttons=QtWidgets.QMessageBox.Ok)
                    return
                cats.append(text)
                cats_probabilities.append(prob)
            if len(set(cats)) != count:
                QtWidgets.QMessageBox.information(self,
                                                  "Повторяющиеся номиналы",
                                                  "Среди номниалов есть одинаковые, "
                                                  "поэтому добавление фактора невозможно",
                                                  buttons=QtWidgets.QMessageBox.Ok)
                return
            factor = NominalFactor(name, role, cats=cats, cats_probabilities=cats_probabilities)
        if not self.cognitive_model.add_factor(factor):
            QtWidgets.QMessageBox.information(self,
                                              "Имя уже занято",
                                              "Имя этого фактора уже занято, выберите другое",
                                              buttons=QtWidgets.QMessageBox.Ok)
        self.notify_main()
