# -*- coding: utf-8 -*-
import pandas as pd

from forms.CM_ImitationForm import Ui_ImitationForm
from logic.CM_Abstracts import AbstractEstimatorLogic


class ImitationWindow(AbstractEstimatorLogic, Ui_ImitationForm):

    def __init__(self, params_dict, cognitive_model):
        super(ImitationWindow, self).__init__(params_dict, cognitive_model)
        self.setupUi(self)
        self.model_changed()
        self.btnSave.clicked.connect(self.save_data)
        self.btnCreate.clicked.connect(self.click_create)

    def model_changed(self):
        self.data = pd.DataFrame()
        self.data_changed()

    def click_create(self):
        M = self.spinBox.value()
        self.create_data(M)
