# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import networkx as nx

from forms.CM_GraphForm import Ui_GraphForm
from logic.CM_Abstracts import AbstractLogic


class GraphWindow(AbstractLogic, Ui_GraphForm):

    def __init__(self, cognitive_model):
        super(GraphWindow, self).__init__(cognitive_model)
        self.setupUi(self)

        self.fig = plt.figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        self.widget.layout().addWidget(self.canvas)

        self.pushButton.clicked.connect(self.draw_graph)

        self.graph = nx.DiGraph()
        self.node_colors = []
        self.node_borders_colors = []
        self.positive_edges = []
        self.negative_edges = []

        self.model_changed()

    def model_changed(self):
        links = self.cognitive_model.get_links()
        factors = set()
        for L in links:
            factors.add(L.factor_to)
            factors.add(L.factor_from)
        edges_list = [L.get_networkx_edge() for L in links]
        self.graph.clear()
        self.graph.add_nodes_from([f.get_id() for f in factors])
        self.graph.add_weighted_edges_from(edges_list)
        self.node_colors = []
        self.node_borders_colors = []
        for f in factors:
            if f.scale == 0:
                self.node_colors.append('#ffc8c8')
                self.node_borders_colors.append('r')
            elif f.scale == 1:
                self.node_colors.append('#c8c8ff')
                self.node_borders_colors.append('b')
            else:
                self.node_colors.append('#c8ffc8')
                self.node_borders_colors.append('g')
        self.positive_edges = list(filter(lambda x: x[2] >= 0, edges_list))
        self.negative_edges = list(filter(lambda x: x[2] < 0, edges_list))

        # Change the graph
        self.draw_graph()

    def draw_graph(self):
        self.fig.clear()
        try:
            if self.comboBox.currentIndex() == 0:
                # Position nodes on a circle
                pos = nx.circular_layout(self.graph)
            elif self.comboBox.currentIndex() == 1:
                # Position nodes in a spiral layout.
                pos = nx.spiral_layout(self.graph)
            elif self.comboBox.currentIndex() == 3:
                # Position nodes without edge intersections
                pos = nx.planar_layout(self.graph)
            else:
                # Position nodes uniformly at random in the unit square.
                pos = nx.random_layout(self.graph)

            # Drawing nodes
            nx.draw_networkx_nodes(self.graph, pos, node_size=800, node_color=self.node_colors,
                                   edgecolors=self.node_borders_colors, linewidths=1.5)
            nx.draw_networkx_labels(self.graph, pos, font_size=14)
            # Then positive links
            width = [edge[2]/20 + 1 for edge in self.positive_edges]
            nx.draw_networkx_edges(self.graph, pos, edgelist=self.positive_edges, width=width, edge_color='r',
                                   arrowsize=20)
            # And negative links
            width = [abs(edge[2] / 20) for edge in self.negative_edges]
            nx.draw_networkx_edges(self.graph, pos, edgelist=self.negative_edges, width=width, edge_color='b',
                                   arrowsize=20, style='dashed')
            weights = nx.get_edge_attributes(self.graph, 'weight')
            # Printing links' strengths
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=weights)
            # Updating canvas
        except (nx.NetworkXException, ValueError):
            # Something can go wrong with this library
            self.fig.clear()
            QtWidgets.QMessageBox.information(self,
                                              "Ошибка библиотеки networkx",
                                              "Не удалось отобразить граф. Попробуйте выбрать другой тип графа "
                                              "или изменить когнитивную модель",
                                              buttons=QtWidgets.QMessageBox.Ok)
        finally:
            self.canvas.draw()
            self.update()
