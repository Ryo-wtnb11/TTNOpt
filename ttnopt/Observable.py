from typing import List, Optional

import numpy as np
from fractions import Fraction


class Observable:
    """A class for observables."""

    def __init__(self, 
                 indices: List[int],
                 operators_list: List[List[str]],
                 coef_list: List[float]):
        """Initialize an Observable object.

        Args:
            indices (List[int]): Indices of the observable.
            operators_list (List[List[str]]): List of operators for each index.
            coef_list (List[float]): Coefficients for each operator.
        """
        self.indices = indices
        self.operators_list = operators_list
        self.coef_list = coef_list

        self.indices_num = len(indices)
        self.operators_num = len(operators_list)


def spin_ind(spin_num):
    if isinstance(spin_num, str):
        if spin_num.startswith("S="):
            spin_value_str = spin_num[2:]
        else:
            raise ValueError("Invalid spin string format. Expected format 'S=...'.")

        if "/" in spin_value_str:
            numerator, denominator = spin_value_str.split("/")
            spin_value = int(numerator) / int(denominator)
        else:
            spin_value = float(spin_value_str)
    elif isinstance(spin_num, (int, float)):
        spin_value = spin_num
    else:
        raise TypeError("Invalid type for spin_num. Expected a string or a number.")

    # スピンの値を分数として取得
    fraction = Fraction(spin_value).limit_denominator()
    numerator, denominator = fraction.numerator, fraction.denominator

    # "S="形式の文字列を返す
    return f"S={numerator}/{denominator}" if denominator != 1 else f"S={numerator}"


def bare_spin_operator(spin, spin_num):
    spin_num = spin_ind(spin_num)
    if spin_num == "S=1/2":
        if spin == "S+":
            return np.array([[0, 2], [0, 0]], dtype=np.float64)
        elif spin == "S-":
            return np.array([[0, 0], [2, 0]], dtype=np.float64)
        elif spin == "Sz":
            return np.array([[1, 0], [0, -1]], dtype=np.float64)
    elif spin_num == "1":
        print("error")


def spin_dof(spin_num):
    spin_num = spin_ind(spin_num)
    spin_value_str = spin_num[2:]
    if "/" in spin_value_str:
        numerator, denominator = spin_value_str.split("/")
        spin_value = int(numerator) / int(denominator)
    else:
        spin_value = float(spin_value_str)
    degrees_of_freedom = int(2 * spin_value + 1)
    return degrees_of_freedom
