# -*- coding: utf-8 -*-
"""Contains main functions to work with cognitive model"""

from random import normalvariate
from typing import Optional, Union

import numpy as np
import pandas as pd
import dbf

from model.CM_classes import AbstractFactor, QuantitativeFactor, OrdinalFactor, NominalFactor

QUALITIES = ("Отсутствует", "Слабая", "Умеренная", "Заметная", "Высокая", "Весьма высокая")


def to_cheddoc(strength: float) -> str:
    """Converts correlation strength to qualitative scale using Cheddoc scale"""
    strength = abs(strength)
    if strength < 0.1:
        return QUALITIES[0]
    elif strength < 0.3:
        return QUALITIES[1]
    elif strength < 0.5:
        return QUALITIES[2]
    elif strength < 0.7:
        return QUALITIES[3]
    elif strength < 0.9:
        return QUALITIES[4]
    else:
        return QUALITIES[5]


def number_2_quality(strength: int) -> int:
    """Converts correlation strength to quality index"""
    if strength > 100:
        return 5
    elif strength < 0:
        return 0
    else:
        return (strength + 10) // 20


def quality_2_number(index: int) -> int:
    """Converts quality index to correlation strength"""
    if index > 5:
        return 100
    elif index < 0:
        return 0
    else:
        return index * 20


def get_regression_params(matrix: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Takes cognitive model in matrix (n*n), returns linear regressions params, dispersions and alpha vector.
    :param matrix: matrix of links strengths (n*n)
    :return: matrix of regression parameters (n*n), vector of dispersions (n), and special vector - alpha (n).

    If alpha is 1, then all normal, else model is not correct: some links strengths are overrated.
    Vector alpha will decrease some too high links.
    """
    # Size of model
    n = matrix.shape[0]
    # In first step correlation between some factors is unknown, bur we know strengths os some links
    correlations = matrix + np.eye(n, n) + matrix.transpose()
    # Special parameter to decrease some too high strengths

    def estimate_params(m, R):
        # There is two steps, because it necessary to estimate correlation between all factors
        S = np.zeros(n)
        b = np.zeros((n, n))
        a = np.ones(n)
        # For every factor in model
        for k in range(n):
            # If it dependent factor
            if m[:, k].any():
                X = (m[:, k] != 0)
                rxy = m[:, k][X]
                Rxx = R[X][:, X]
                bxy = rxy.dot(np.linalg.inv(Rxx))
                S[k] = 1 - rxy.dot(np.linalg.inv(Rxx)).dot(rxy)
                if S[k] < 0:
                    a[k] = min(a[k], abs(0.99999 / (S[k] - 1)))
                b[:, k][X] = bxy
            # Decrease params or don't change
            b[:, k] = b[:, k] * (a[k] ** 0.5)
            S = (S - 1) * a[k] + 1
        return b, S, a

    # First step - estimate parameters, when not all correlations is known
    regression, dispersions, alpha = estimate_params(matrix, correlations)
    # Making imitation data to check correlations
    data = pd.DataFrame(np.random.uniform(-3.25, 3.25, (5000, n)))
    for j in range(1000):
        next_record = data.iloc[j]
        for i in range(n):
            if matrix[:, i].any():
                next_record[i] = np.dot(next_record, regression[:, i]).sum() + normalvariate(0, dispersions[i] ** 0.5)
        data.iloc[j] = next_record
    # Estimate correlations
    correlations = np.array(data.corr().fillna(0))
    correlations[range(n), range(n)] = np.ones((1, n))
    # Estimate parameters again with full correlation matrix
    regression, dispersions, alpha = estimate_params(matrix, correlations)
    return regression, dispersions, alpha


def normalize(value: any, factor: AbstractFactor) -> float:
    """Takes the value of factor and return float number between -3.25 and 3.25"""
    if factor.scale == 0:
        factor: QuantitativeFactor
        return (value - (factor.max_value + factor.min_value) / 2) / (factor.max_value - factor.min_value) * 6.5
    elif factor.scale == 1:
        factor: OrdinalFactor
        value = round(value)
        cum_sum = [0] + factor.cum_sum
        return ((cum_sum[value - 1] + cum_sum[value]) / 2 - 0.5) * 6.5
    else:
        factor: NominalFactor
        i = factor.cats.index(value)
        cum_sum = [0] + factor.cum_sum
        return ((cum_sum[i] + cum_sum[i + 1]) / 2 - 0.5) * 6.5


def reverse_normalize(value: float, factor: AbstractFactor) -> float:
    """Takes float number between -3.25 and 3.25 and return the value of factor"""
    value = min(max(value, -3.25), 3.25)
    if factor.scale == 0:
        factor: QuantitativeFactor
        ans = value / 6.5 * (factor.max_value - factor.min_value) + (factor.max_value + factor.min_value) / 2
        return round(ans, 3)
    elif factor.scale == 1:
        factor: OrdinalFactor
        p = value / 6.5 + 0.5
        for i in range(factor.max_value):
            if p < factor.cum_sum[i]:
                return i + 1
        return factor.max_value
    else:
        factor: NominalFactor
        p = value / 6.5 + 0.5
        for i in range(factor.count):
            if p < factor.cum_sum[i]:
                return factor.cats[i]
        return factor.cats[-1]


def get_graph_visiting_order(regression: np.ndarray) -> list:
    """Takes matrix of regression parameters, returns list of factors' indexes in order of visiting."""
    n = regression.shape[0]
    order = []
    current_wave = []
    for i in range(n):
        if not regression[:, i].any():
            current_wave.append(i)
    if not len(current_wave):
        current_wave.append(np.random.randint(0, n))
    k = 0
    threshold = min(1000, n * n / 2)
    while current_wave and k < threshold:
        next_wave = set()
        for i in current_wave:
            next_wave |= set(regression[i, :].nonzero()[0])
        order += current_wave
        current_wave = list(next_wave)
        k += len(next_wave)
    order += current_wave
    return order


def dict_state_2_data_frame(state: dict[str, any], factors: tuple[AbstractFactor]) -> Optional[pd.DataFrame]:
    """Creates pandas data frame from data dictionary using factors.
    If there is wrong data, raising error

    :param state: dict of values of factors (key is factor name, values could be int, float and str types)
    :param factors: tuple of factors
    :return: data frame of state or None
    """
    for f in factors:
        if f.name in state:
            if not f.is_in_interval(state[f.name]):
                raise ValueError('Value %s not in intervals' % state[f.name])
            state[f.name] = [state[f.name]]
        else:
            state[f.name] = [f.generate_value()]
    data = pd.DataFrame(state)
    return data


def make_imitation_data(regression: np.ndarray, dispersions: np.ndarray, factors: tuple[AbstractFactor],
                        visiting_order: list, M: int, fixed_controls: bool = False, fixed_observables: bool = False,
                        state: Optional[dict[str, any]] = None) -> pd.DataFrame:
    """
    Method to create imitation data, when observation data is not available.
    Every record in result is final state of system (system could be unstable if there is cycle(s)!).
    Warning: Be careful with nominal factors! In this method they are ordered, but in fact it is strong simplification.
    Use nominals in this method ONLY when there is no any observations data.
    May be slow in models with cycles with high volume of data.

    :param regression: numpy array (n*n) of regression parameters
    :param dispersions: numpy vector (n) of dispersions
    :param factors: tuple of factors
    :param M: needed volume of data (number of records) between 10 and 10000
    :param visiting_order: list of factors indexes in tuple, order of factors to estimate their values
    :param fixed_controls: True if controlled factors should not change, else False
    :param fixed_observables: True if observable factors should not change, else False
    :param state: if one of two flags are True then needed dictionary in form "factor_name: value"
    :return: returns pandas data frame of imitation data, where columns are factors names
    """
    deviations = (dispersions ** 0.5)
    M = min(max(M, 10), 10000)
    n = len(factors)
    if state:
        state = dict_state_2_data_frame(state, factors)
        for f in factors:
            state[f.name] = state[f.name].apply(lambda x: normalize(x, f))
    data = pd.DataFrame(np.random.uniform(-3.25, 3.25, (M, n)), columns=[f.name for f in factors])
    for k in visiting_order:
        if factors[k].role == 0 and fixed_controls:
            data[factors[k].name] = state[factors[k].name].iloc[0]
        elif factors[k].role == 1 and fixed_observables:
            data[factors[k].name] = state[factors[k].name].iloc[0]
        if regression[:, k].any():
            data[factors[k].name] = data.apply(lambda row:
                                               np.dot(row, regression[:, k]).sum() + normalvariate(0, deviations[k]),
                                               axis=1)
    for f in factors:
        data[f.name] = data[f.name].apply(lambda x: reverse_normalize(x, f))
    return data


def estimate_transient_response(regression: np.ndarray, factors: tuple[AbstractFactor],
                                state0: dict[str, any], state1: dict[str, any]) -> pd.DataFrame:
    """
    Method to estimate transient response of system.
    Warning! Doesn't work correctly with nominals factors!
    Difference with imitation data in lack of variance and every record in frame is a step of process (not final state).
    If there is no inputs then process starts with random factor.

    :param regression: numpy array (n*n) of regressions parameters
    :param factors: tuple of factors
    :param state0: dictionary in form {factor_name: value}, state of system at step 0
    :param state1: dictionary in form {factor_name: value}, state of system at step 1
    :return: pandas data frame of transient response
    """
    n = len(factors)
    current_wave = []
    # First state
    data = dict_state_2_data_frame(state0, factors)
    data = data.append(dict_state_2_data_frame(state1, factors), ignore_index=True)
    for f in factors:
        data[f.name] = data[f.name].apply(lambda x: normalize(x, f))
    # Starts with inputs
    for i in range(n):
        if not regression[:, i].any():
            current_wave.append(i)
    # Else with random factor
    if not len(current_wave):
        current_wave.append(np.random.randint(0, n))
    j = 0
    # End when there is no more factors to go, or too long process with cycles
    while current_wave and j < 100:
        next_wave = set()
        # Identify the next factors to visit
        for i in current_wave:
            next_wave |= set(regression[i, :].nonzero()[0])
        # Take the last state
        last_state = pd.Series(data.iloc[-1], copy=True)
        # And calculating next wave
        for k in next_wave:
            last_state[factors[k].name] = np.dot(last_state, regression[:, k]).sum()
        data = data.append(last_state, ignore_index=True)
        # If the process diverges in time or there is no changes in waves then break
        E = (abs(data.iloc[-1] - data.iloc[-2])).max()
        if last_state.max() > 3.25 or last_state.min() < -3.25 or E < 0.01:
            break
        current_wave = next_wave
        j += 1
    for f in factors:
        data[f.name] = data[f.name].apply(lambda x: reverse_normalize(x, f))
    return data


def estimate_transient_response_normalized(regression: np.ndarray, factors: tuple[AbstractFactor],
                                           state: dict[str, any]) -> pd.DataFrame:
    """
    The same as estimate_transient_response, but without reverse normalizing at the end,
    and normalizing at the beginning

    :param regression: numpy array (n*n) of regressions parameters
    :param factors: tuple of factors
    :param state: dictionary in form {factor_name: value}, state of system at step 0
    :return: pandas data frame of transient response
    """
    n = len(factors)
    current_wave = []
    # First state
    data = pd.DataFrame({f.name: [state[f.name]] for f in factors})
    # Starts with inputs
    for i in range(n):
        if not regression[:, i].any():
            current_wave.append(i)
    # Else with random factor
    if not len(current_wave):
        current_wave.append(np.random.randint(0, n))
    j = 0
    # End when there is no more factors to go, or too long process with cycles
    while current_wave and j < 100:
        next_wave = set()
        # Identify the next factors to visit
        for i in current_wave:
            next_wave |= set(regression[i, :].nonzero()[0])
        # Take the last state
        last_state = pd.Series(data.iloc[-1], copy=True)
        # And calculating next wave
        for k in next_wave:
            last_state[factors[k].name] = np.dot(last_state, regression[:, k]).sum()
        data = data.append(last_state, ignore_index=True)
        # If the process diverges in time or there is no changes in waves then break
        E = (abs(data.iloc[-1] - data.iloc[-2])).max()
        if last_state.max() > 3.25 or last_state.min() < -3.25 or E < 0.01:
            break
        current_wave = next_wave
        j += 1
    return data


def data_2_dbf(data: pd.DataFrame, factors: tuple[AbstractFactor], file_path: str):
    """Takes data frame, tuple of factors and saves data to selected data base file."""
    M = data.shape[0]
    # Creating columns and data types
    fields = ''
    for f in factors:
        if f.scale == 0:
            fields += f.get_id() + ' N(15,4);'
        elif f.scale == 1:
            fields += f.get_id() + ' N(5, 0);'
        else:
            fields += f.get_id() + ' C(50);'
    db = dbf.Table(file_path, fields, codepage='cp866')
    db.open(dbf.DbfStatus.READ_WRITE)
    # Saving data to database
    for i in range(0, M):
        db.append(tuple(data.iloc[i].tolist()))
    db.close()


def estimate_prediction(data: pd.DataFrame, factors: tuple[AbstractFactor], p: float = 0.95) -> list[tuple, dict]:
    """
    Takes data frame, tuple of factors. Returns list of predictions.
    The larger the amount of data, the more accurate the prediction estimates.
    The result list contains tuple for each factor:
    For quantitative factor it is confidence interval and mean estimations:
    (left, mean, right)
    For ordinal factors it is confidence interval and median estimations:
    (left, median, right)
    For nominal factors it is categories occurrence frequency in dict:
    dict(cat1: freq1, cat2: freq2,... catN: freqN)

    :param data: pandas data frame of observation (imitation) data
    :param factors: tuple of factors in model
    :param p: confidence level for confidence interval between 0.75 and 1
    :return: list of tuples of estimating params.
    """
    p = max(min(0.999, p), 0.75)
    ans = []
    for f in factors:
        if f.scale == 0:
            left = round(data[f.name].quantile(q=((1-p)/2)), 3)
            right = round(data[f.name].quantile(q=((1+p)/2)), 3)
            mean = round(data[f.name].mean(), 3)
            ans.append((left, mean, right))
        elif f.scale == 1:
            left = round(data[f.name].quantile(q=((1-p)/2)))
            right = round(data[f.name].quantile(q=((1+p)/2)))
            median = round(data[f.name].median())
            ans.append((left, median, right))
        elif f.scale == 2:
            count = data[f.name].value_counts().to_dict()
            ans.append(count)
    return ans


def compare_values(obtained_value: any, desired_value: any, factor: AbstractFactor) -> float:
    """Takes obtained and desired values of targeted factor, returns criterion value"""
    if factor.scale == 0:
        factor: QuantitativeFactor
        ans = (obtained_value - desired_value) / (factor.max_value - factor.min_value)
        return round(ans, 3)
    elif factor.scale == 1:
        factor: OrdinalFactor
        ans = (obtained_value - desired_value) / (factor.max_value - 1)
        return round(ans, 3)
    else:
        factor: NominalFactor
        return obtained_value != desired_value


def criterion(factors: tuple[AbstractFactor], weights: dict[str, float], targets: dict[str, any],
              data: Union[pd.DataFrame, pd.Series]) -> float:
    """
    Takes tuple of factors, weights of targeted factors, dict of required values of targeted factors.
    Calculating special float number - the criterion of proximity values of targeted factors from data to the
    desired values. If criterion is close to 0, then data is close to desired. The larger the criterion,
    the greater the difference between the obtained and desired values.

    :param factors: tuple of factors in model
    :param weights: dictionary {factor_name: weight}, where 'weight' is float number, that shows priority of factor
    :param targets: dictionary {factor_name: target}, where 'target' is desired value of targeted factor
    :param data: pandas data frame of observation (imitation) data
    :return: float positive number, shows how values of targeted factors from data is close to desired values
    """
    E = 0
    """Only for targeted factors"""
    if isinstance(data, pd.DataFrame):
        for f in factors:
            if f.role == 3:
                E += abs(data[f.name].apply(lambda x: compare_values(x, targets[f.name], f)) ** 2).mean() \
                     * weights[f.name]
    else:
        for f in factors:
            if f.role == 3:
                E += compare_values(data[f.name], targets[f.name], f) ** 2 * weights[f.name]
    return round(E * (3.25 ** 2), 3)


def gradient_method(regression: np.ndarray, factors: tuple[AbstractFactor], weights: dict[str, float],
                    targets: dict[str, any], current_state: dict[str, any], alpha: float = 1,
                    beta: float = 0.25) -> dict[str, any]:
    """
    Gradient descent method for evaluating the values of control actions,
    allowing to achieve the desired values of controlled factors.

    :param regression: numpy array (n*n) of regressions parameters
    :param factors: tuple of factors in model
    :param weights: dictionary {factor_name: weight}, where 'weight' is float number, that shows priority of factor
    :param targets: dictionary {factor_name: target}, where 'target' is desired value of targeted factor
    :param current_state: dictionary in form {factor_name: value}, state of system at step 0
    :param alpha: float positive, the rate of algorithm
    :param beta: float positive or zero, the inertia of algorithm
    :return: dictionary {factor_name: result}, where 'result' is recommended value of factors
    """
    controlled = [f for f in factors if f.role == 0]
    targeted = [f for f in factors if f.role == 3]
    targets = targets.copy()
    # Estimate the influence of inputs on targeted factors
    influence_on_targeted = {}
    for u in controlled:
        # All zeros except for one controlled input, where the impact is sent
        clear_state = {f.name: 0.0 for f in factors}
        clear_state[u.name] = 1.0
        response = estimate_transient_response_normalized(regression, factors, clear_state).iloc[-1]
        influence_on_targeted[u.name] = response
    for f in factors:
        current_state[f.name] = normalize(current_state[f.name], f)
        if f.name in targets:
            targets[f.name] = normalize(targets[f.name], f)
    current_response = estimate_transient_response_normalized(regression, factors, current_state).iloc[-1]
    E = sum([(targets[key] - current_response[key]) ** 2 for key in targets.keys()])
    k = 0
    previous_state = current_state.copy()
    while k < 1000 and E > 0.01:
        for u in controlled:
            current_state[u.name] += (current_state[u.name] - previous_state[u.name]) * beta
            for y in targeted:
                next_value = current_state[u.name] - (current_response[y.name] - targets[y.name]) \
                             * influence_on_targeted[u.name][y.name] * weights[y.name] * alpha
                current_state[u.name] = min(max(next_value, -3.25), 3.25)
        current_response = estimate_transient_response_normalized(regression, factors, current_state).iloc[-1]
        E = sum([(targets[key] - current_response[key]) ** 2 for key in targets.keys()])
        previous_state = current_state.copy()
        k += 1
    for f in factors:
        current_state[f.name] = reverse_normalize(current_state[f.name], f)
    return current_state
