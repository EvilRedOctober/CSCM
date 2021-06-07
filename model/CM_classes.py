# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from random import normalvariate, choice, random

import pandas as pd
import numpy as np

# Basics values, that can exists
POSSIBLE_MAX = 9999999
POSSIBLE_MIN = -9999999

# Using terms
SCALES = ("Количественный", "Порядковый", "Номинальный")
ROLES = ("Управляемый", "Наблюдаемый", "Прочий", "Целевой")
SIGNS_OF_ROLES = ('U', 'Z', 'X', 'Y')


class AbstractFactor(ABC):
    """Abstract class for factors."""

    def __init__(self, name: str, scale: int, role: int, number: int):
        """
        An implementation of factor, could have one of three scales and one of four roles, that changing the
        logic of it's work.

        :param name: any text
        :param scale: 0, 1 or 2 (Quantitative, Ordinal or Nominal)
        :param role: 0, 1, 2 or 3 (Controlled, Observable, Other or Targeted)
        """
        self.name = name
        self.number = number
        self.scale = scale
        self.role = role

    def change_number(self, number: int):
        """When factor change position in model, it's necessary to change it number."""
        self.number = number

    def __repr__(self):
        return SIGNS_OF_ROLES[self.role] + str(self.number) + ' ' + self.name + ' ' + SCALES[self.scale] + ' ' \
               + ROLES[self.role]

    def get_id(self) -> str:
        """Return short name for factor."""
        return SIGNS_OF_ROLES[self.role] + str(self.number)

    @abstractmethod
    def generate_value(self) -> any:
        """Generate a random value."""
        return None

    @abstractmethod
    def is_in_interval(self, value) -> bool:
        """Check that the value belongs to the range of possible values of the factor."""
        return False

    def encode_json(self) -> dict:
        """Return dict of __slots__ to save factor into json format."""
        d = {attr: self.__getattribute__(attr) for attr in self.__slots__ if attr != 'cum_sum'}
        return d


class QuantitativeFactor(AbstractFactor):
    """Factor whose values can be float numbers"""

    __slots__ = "name", "number", "scale", "role", "max_value", "min_value", "mean", "standard_deviation"

    def __init__(self, name: str, role: int, number: int = 0,
                 max_value: float = 100, min_value: float = 0, mean: float = None, standard_deviation: float = None):
        """
        A quantitative factor, with numeric value

        :param name: any text
        :param role: 0, 1, 2 or 3 (Controlled, Observable, Other or Targeted)
        :param number: a natural number, index of factor
        :param max_value: the values of factor will never be grater this number
        :param min_value: the values of factor will never be lower this number
        :param mean: float number between max and min values, describes the center of the distribution
        :param standard_deviation: float positive number, describes the range of the distribution
        """

        super().__init__(name, 0, role, number)
        self.max_value = max(min(max_value, POSSIBLE_MAX), POSSIBLE_MIN + 1)
        self.min_value = max(min(min_value, self.max_value - 1), POSSIBLE_MIN)
        if (mean is None) or (mean >= self.max_value or mean <= self.min_value):
            self.mean = (self.max_value + self.min_value) / 2
        else:
            self.mean = mean
        if standard_deviation is None or standard_deviation < (self.max_value - self.min_value) / 6:
            self.standard_deviation = (self.max_value - self.min_value) / 6
        else:
            self.standard_deviation = standard_deviation

    def generate_value(self) -> float:
        """Generate a random value."""
        ans = normalvariate(self.mean, self.standard_deviation)
        ans = max(min(ans, self.max_value), self.min_value)
        return ans

    def is_in_interval(self, value: float) -> bool:
        """Check that the value belongs to the range of possible values of the factor."""
        return self.min_value <= value <= self.max_value


class OrdinalFactor(AbstractFactor):
    """Factor whose values can be ranks"""

    __slots__ = "name", "number", "scale", "role", "max_value", "ranks_probabilities", "cum_sum"

    def __init__(self, name: str, role: int, number: int = 0,
                 max_value: int = 2, ranks_probabilities: list[float, ...] = ()):
        """
        An ordinal factor, with ranks value

        :param name: any text
        :param role: 0, 1, 2 or 3 (Controlled, Observable, Other or Targeted)
        :param number: a natural number, index of factor
        :param max_value: a max rank (the min is always 1)
        :param ranks_probabilities: ranks occurrence probabilities
        """

        super().__init__(name, 1, role, number)
        self.max_value = max(min(max_value, POSSIBLE_MAX), 1)
        if not ranks_probabilities:
            ranks_probabilities = [1 / max_value for _ in range(1, self.max_value + 1)]
        elif len(ranks_probabilities) != self.max_value:
            raise ValueError('Length of ranks probabilities must be equal to max value!')
        else:
            s = sum(ranks_probabilities)
            if abs(s - 1) > 0.01:
                ranks_probabilities = [prob/s for prob in ranks_probabilities]
        self.ranks_probabilities = ranks_probabilities.copy()
        self.cum_sum = np.cumsum(self.ranks_probabilities).tolist()

    def generate_value(self) -> float:
        """Generate a random value."""
        p = random()
        for i in range(self.max_value):
            if p < self.cum_sum[i]:
                return i + 1
        return self.max_value

    def is_in_interval(self, value: int) -> bool:
        """Check that the value belongs to the range of possible values of the factor."""
        return 1 <= value <= self.max_value


class NominalFactor(AbstractFactor):
    """Factor whose values can be one of certain categories"""

    __slots__ = "name", "number", "scale", "role", "cats", "cats_probabilities", "cum_sum"

    def __init__(self, name: str, role: int, number: int = 0,
                 cats: list[str, ...] = ('A', 'B', 'C', 'D'), cats_probabilities: list[float, ...] = ()):
        """
        A nominal factor with list of categories

        :param name: any text
        :param role: 0, 1, 2 or 3 (Controlled, Observable, Other or Targeted)
        :param number: a natural number, index of factor
        :param cats: a tuple of possible categories, example: ('A', 'B', 'C')
        :param cats_probabilities: a tuple (or list) of probabilities for categories
        """

        super().__init__(name, 2, role, number)
        self.count = len(cats)
        if not cats_probabilities:
            cats_probabilities = [1 / self.count for _ in range(1, self.count + 1)]
        elif len(cats_probabilities) != len(cats):
            raise ValueError('Length of categories probabilities must be equal to max value!')
        else:
            s = sum(cats_probabilities)
            if abs(s - 1) > 0.01:
                cats_probabilities = [prob/s for prob in cats_probabilities]
        self.cats_probabilities = cats_probabilities.copy()
        self.cum_sum = np.cumsum(self.cats_probabilities).tolist()
        self.cats = cats.copy()

    def generate_value(self) -> str:
        """Generate a random value."""
        p = random()
        for i in range(self.count):
            if p < self.cum_sum[i]:
                return self.cats[i]
        return self.cats[-1]

    def is_in_interval(self, value: any) -> bool:
        """Check that the value belongs to the range of possible values of the factor."""
        return value in self.cats


class InterfactorLink:
    """Have a sign and strength"""

    __slots__ = "factor_from", "factor_to", "strength"

    def __init__(self, factor_from: AbstractFactor, factor_to: AbstractFactor, strength: float):
        """
        A link between two factors, the and the strength of influences

        :param factor_from: An influential factor
        :param factor_to: A dependent factor
        :param strength: correlation between -1 and 1 (for nominal only positive)
        """
        self.factor_from = factor_from
        self.factor_to = factor_to
        if factor_from == factor_to:
            raise ValueError('Loops are prohibited')
        if factor_to.role < 2:
            raise ValueError('Cannot connect to input')
        scale = max(factor_from.scale, factor_to.scale)
        if scale == 2:
            # Nominal factors don't have sign of link,
            # because it's impossible to say, that nominal is rising or decreasing
            self.strength = min(max(strength, 0), 1)
        else:
            self.strength = min(max(strength, -1), 1)

    def is_incident(self, factor) -> bool:
        return self.factor_from == factor or self.factor_to == factor

    def get_factors_numbers(self) -> tuple[int, int]:
        return self.factor_from.number, self.factor_to.number

    def get_networkx_edge(self) -> tuple[str, str, int]:
        return self.factor_from.get_id(), self.factor_to.get_id(), round(self.strength*100)

    def __repr__(self):
        return "%s==(%.f%%)=>%s" % (self.factor_from.get_id(), self.strength * 100, self.factor_to.get_id())

    def encode_json(self) -> dict:
        d = {"factor_to": self.factor_to.number,
             "factor_from": self.factor_from.number,
             "strength": self.strength}
        return d


# Basic class for a Cognitive model
class CognitiveModel:
    """A representation of control system in the form of a cognitive model."""
    __slots__ = "name", "description", "__list_factors", "__available_number", "__dict_links", \
                "__matrix_of_links"

    def __init__(self, name: str, description: str):
        """
        A cognitive model of control system with lots of functions.
        Can be presented in table or graph form (factors - nodes, links - edges).

        :param name: Any text
        :param description: Any text
        """
        self.name = name
        self.description = description
        self.__list_factors = []
        self.__available_number = 0
        self.__dict_links = {}
        self.__matrix_of_links = np.array([])

    def reset_values(self):
        """Any changes in model require to clear current matrix of links."""
        self.__matrix_of_links = np.array([])

    def add_factor(self, factor) -> bool:
        """Adding a factor with specific parameters. The model numbers start from 0, but for factors it is 1."""
        if factor.name in self.__dict_links:
            return False
        self.__available_number += 1
        factor.change_number(self.__available_number)
        self.__list_factors.append(factor)
        self.__dict_links[factor.name] = []
        # Factors changed! And data must be removed
        self.reset_values()
        return True

    def add_link(self, factor_from: AbstractFactor, factor_to: AbstractFactor, strength: float):
        """To add interfactor link."""
        link = InterfactorLink(factor_from, factor_to, strength)
        self.del_existing_link(factor_from, factor_to)
        self.__dict_links[factor_from.name].append(link)
        self.reset_values()

    def del_factor(self, factor: AbstractFactor):
        """To delete factor."""
        # The program number start from 0
        number = factor.number
        factor = self.__list_factors.pop(number - 1)
        # Delete links to factor
        for factor_from in self.__list_factors:
            for link in self.__dict_links[factor_from.name]:
                if factor == link.factor_to:
                    self.del_link(link)
        # Delete links from factor
        del self.__dict_links[factor.name]
        # Change numbers for other factors
        for i, factor in enumerate(self.__list_factors[number - 1:]):
            # But for user numbers start from 1
            factor.change_number(i + number)
        self.__available_number -= 1
        # Factors changed! And data must be removed
        self.reset_values()

    def del_factor_by_numbers(self, a: int):
        """To delete factor, when known only its number."""
        factor = self.__list_factors[a]
        self.del_factor(factor)

    def del_link(self, link: InterfactorLink):
        """To delete interfactor link."""
        self.__dict_links[link.factor_from.name].remove(link)
        self.reset_values()

    def del_existing_link(self, factor_from: AbstractFactor, factor_to: AbstractFactor):
        """a - number factor from, b - number factor to."""
        for link in self.__dict_links[factor_from.name]:
            if link.factor_to.name == factor_to.name:
                self.del_link(link)
                return

    def get_factors(self):
        """Return all factors."""
        return tuple(self.__list_factors)

    def get_links(self):
        """Return all links."""
        ans = []
        for factor in self.__list_factors:
            ans += self.__dict_links[factor.name]
        return tuple(ans)

    def clear_factors(self):
        """Delete all factors (and links too)."""
        self.reset_values()
        self.__list_factors = []
        self.__available_number = 0
        self.__dict_links = {}

    def clear_links(self):
        """Delete all links."""
        self.reset_values()
        self.__dict_links = {factor.name: [] for factor in self.__list_factors}

    def sort_factors(self):
        """Using topological sorting for factors."""
        if self.check_cycles():
            raise ValueError('There is cycle! Cannot sort')
        # First - get inputs
        controlled, observable, others = [], [], []
        for factor in self.__list_factors:
            if factor.role == 0:
                controlled.append(factor)
            elif factor.role == 1:
                observable.append(factor)
            else:
                others.append(factor)

        # Then - using topological sorting
        checked = set()
        reverse_factors = []

        def dfs_topological(v):
            checked.add(v)
            for link in self.__dict_links[v.name]:
                w = link.factor_to
                if w not in checked:
                    dfs_topological(w)
            reverse_factors.append(v)

        for factor in others:
            if factor not in checked:
                dfs_topological(factor)
        # Getting all together
        self.__list_factors = controlled + observable + reverse_factors[-1::-1]
        # Changing the numbers
        [factor.change_number(i + 1) for i, factor in enumerate(self.__list_factors)]
        # Order of factors changed! And data must be removed
        self.reset_values()

    def check_cycles(self) -> bool:
        """Return True if no cycles in graph, else False."""
        grey = set()
        black = set()

        def dfs(v):
            grey.add(v)
            for link in self.__dict_links[v]:
                w = link.factor_to.name
                if w in black:
                    continue
                elif w in grey:
                    return True
                elif dfs(w):
                    return True
            black.add(v)
            return False

        for factor in self.__list_factors:
            if dfs(factor.name):
                return True
        return False

    def check_connectivity(self) -> bool:
        """Return True if connected graph, else False."""
        reached = set()
        # Working with undirected graph
        n = len(self.__list_factors)
        matrix = self.get_matrix_of_links()
        to_check = [choice(self.__list_factors).number - 1]
        # Then just checking all nodes
        while to_check:
            j = to_check.pop()
            reached.add(j)
            to_check = to_check + [i for i in range(n) if (matrix[i][j] or matrix[j][i]) and i not in reached]
        return len(reached) == n

    def get_matrix_of_links(self) -> np.array:
        """Returns 2D array (matrix) of links in model."""
        if self.__matrix_of_links.size > 0:
            return self.__matrix_of_links
        n = len(self.__list_factors)
        matrix = np.zeros((n, n), dtype="float16")
        for factor_from in self.__list_factors:
            for link in self.__dict_links[factor_from.name]:
                factor_to = link.factor_to
                matrix[factor_from.number - 1][factor_to.number - 1] = link.strength
        self.__matrix_of_links = matrix
        return matrix

    def encode_json(self) -> dict:
        """Return dict with next format:\n
        "CognitiveModel": True - (special flag to load model from this dict).\n
        "name": str - name of model.\n
        "description": str - description of model.\n
        "list_factors": list[AbstractFactor] - list of factors.\n
        "matrix_of_links": list[list] - list of lists, where i, j element is strength of link between i and j factors.
        """
        d = {"CognitiveModel": True, "name": self.name, "description": self.description,
             "list_factors": [factor.encode_json() for factor in self.__list_factors],
             "matrix_of_links": self.get_matrix_of_links().tolist()}
        return d

    def load_json(self, d: dict):
        """Taking special dict (see encode_json) and load model from it."""
        self.clear_factors()
        if "CognitiveModel" not in d:
            raise ValueError('Wrong dictionary! Please, check that the selected json file is correct')
        self.name, self.description = d["name"], d["description"]
        for dict_factor in d["list_factors"]:
            scale = dict_factor.pop("scale")
            if scale == 0:
                factor = QuantitativeFactor(**dict_factor)
            elif scale == 1:
                factor = OrdinalFactor(**dict_factor)
            else:
                factor = NominalFactor(**dict_factor)
            self.add_factor(factor)
        matrix = d["matrix_of_links"]
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if matrix[i][j]:
                    self.add_link(self.__list_factors[i], self.__list_factors[j], matrix[i][j])

    def __repr__(self):
        return self.name + "\n" + self.description + "\n" \
               + ", ".join((str(factor) for factor in self.__list_factors)) \
               + "\n" + ", ".join(" ".join(map(str, self.__dict_links[factor.name])) for factor in self.__list_factors)


if __name__ == '__main__':
    import json

    cm = CognitiveModel('Тестовая модель', 'Описание модели...')
    print(cm)
    factor1 = QuantitativeFactor('Финансирование', 0, max_value=1000000, min_value=50000)
    factor2 = OrdinalFactor('Качество продукции', 2, max_value=10)
    factor3 = NominalFactor('Заключение', 3, cats=['A', 'B', 'C'])
    print(factor1, factor2, factor3)
    cm.add_factor(factor2)
    cm.add_factor(factor3)
    cm.add_factor(factor1)
    f1, f2, f3 = cm.get_factors()
    cm.add_link(f3, f1, .75)
    cm.add_link(f1, f2, .90)
    cm.add_link(f2, f1, .50)
    matr = cm.get_matrix_of_links().__copy__()
    print(cm)
    print("Is there any cycles?", cm.check_cycles())
    print("Is it connected?", cm.check_connectivity())
    print(cm.get_matrix_of_links())
    print('\nSorting...')
    # cm.sort_factors()
    print(cm)
    print("Is there any cycles?", cm.check_cycles())
    print("Is it connected?", cm.check_connectivity())
    print(cm.get_matrix_of_links())
    print('\nDeleting cycle...')
    L = cm.get_links()[0]
    print(L)
    cm.del_link(L)
    print(cm)
    print("Is there any cycles?", cm.check_cycles())
    print("Is it connected?", cm.check_connectivity())
    print(cm.get_matrix_of_links())
    print('\nDeleting other link...')
    L = cm.get_links()[1]
    cm.del_link(L)
    print(cm)
    print("Is there any cycles?", cm.check_cycles())
    print("Is it connected?", cm.check_connectivity())
    print(cm.get_matrix_of_links())
    print('\nRestoring the links and removing factor')
    cm.add_link(f1, f2, -.90)
    cm.add_link(f2, f1, .50)
    cm.del_factor(f1)
    print(cm)
    print("Is there any cycles?", cm.check_cycles())
    print("Is it connected?", cm.check_connectivity())
    print(cm.get_matrix_of_links())
    print('cats_probabilities', factor3.cats_probabilities)
    print('ranks_probabilities', factor2.ranks_probabilities)
    print('factor3.__slots__', factor3.__slots__)
    L = InterfactorLink(factor1, factor2, 75)

    cm = CognitiveModel('Тестовая модель', 'Описание модели...')
    factor1 = QuantitativeFactor('Финансирование', 0, max_value=1000000, min_value=50000)
    factor2 = OrdinalFactor('Качество продукции', 2, max_value=10)
    factor3 = NominalFactor('Заключение', 3, cats=['A', 'B', 'C'])
    cm.add_factor(factor2)
    cm.add_factor(factor3)
    cm.add_factor(factor1)
    f1, f2, f3 = cm.get_factors()
    cm.add_link(f3, f1, -.75)
    cm.add_link(f1, f2, .90)
    cm.add_link(f2, f1, .50)
    json.dump(cm.encode_json(), open("../examples/CM1.json", 'w'), indent=4, ensure_ascii=False)
    json_cm = json.load(open("../examples/CM1.json", 'r'))
    print(json_cm)
    new_cm = CognitiveModel('Энергетическая система', 'Данная модель представляет собой зависимость между факторами, '
                                                      'описывающими состояние города, состояния окружающей среды и '
                                                      'потреблением энергии.\nМодель является тестовой и не претендует '
                                                      'на состоятельность.')
    factors = [QuantitativeFactor("Производство энергии, млн. кВт*ч", 3, max_value=60, min_value=0),
               QuantitativeFactor("Цена энергии, руб./кВт*ч", 2, max_value=10, min_value=1),
               OrdinalFactor("Число заводов, десятки", 2, max_value=10),
               QuantitativeFactor("Потребление энергии, млн. кВт*ч", 3, max_value=50, min_value=5),
               OrdinalFactor("Качество окружающей среды, баллы", 3, max_value=10),
               QuantitativeFactor("Численность населения, млн.", 2, max_value=13, min_value=1),
               QuantitativeFactor("Уровень занятости, %.", 2, max_value=80, min_value=20)]
    [new_cm.add_factor(f) for f in factors]
    new_cm.add_link(factors[0], factors[1], -.70)
    new_cm.add_link(factors[0], factors[2], .50)
    new_cm.add_link(factors[3], factors[0], .80)
    new_cm.add_link(factors[2], factors[3], .90)
    new_cm.add_link(factors[1], factors[3], -.20)
    new_cm.add_link(factors[3], factors[1], .30)
    new_cm.add_link(factors[3], factors[4], -.60)
    new_cm.add_link(factors[5], factors[3], .70)
    new_cm.add_link(factors[4], factors[5], .40)
    new_cm.add_link(factors[3], factors[6], .60)
    new_cm.add_link(factors[6], factors[5], .30)

    json.dump(new_cm.encode_json(), open(r"../examples/Energy System.json", 'w'), indent=4, ensure_ascii=False)
    print(new_cm)
    print(cm)
