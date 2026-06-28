"""ACSM cycle ergometer metabolic equations for VO2 prediction."""

from typing import Literal

import numpy as np
import sympy


class Bike:
    """
    ACSM cycle ergometer metabolic equations for VO2 prediction.

    This class implements the American College of Sports Medicine (ACSM) metabolic
    equations for predicting oxygen consumption during cycle ergometer exercise.
    Separate equations are used for males and females.

    Attributes
    ----------
    VO2 : sympy.Symbol
        Oxygen consumption (ml/kg/min) symbolic variable.
    WEIGHT : sympy.Symbol
        Body weight (kg) symbolic variable.
    POWER : sympy.Symbol
        Cycling power output (watts) symbolic variable.
    males_eq : sympy.Equality
        ACSM equation for males: VO2 = 10.7712*power/weight + 3.5
    females_eq : sympy.Equality
        ACSM equation for females: VO2 = 10.098*power/weight + 3.5
    """

    VO2 = sympy.Symbol("VO2", real=True)
    WEIGHT = sympy.Symbol("WEIGHT", real=True)
    POWER = sympy.Symbol("POWER", real=True)
    males_eq = sympy.Equality(VO2, 10.7712 * POWER / WEIGHT + 3.5)  # type: ignore
    females_eq = sympy.Equality(VO2, 10.098 * POWER / WEIGHT + 3.5)  # type: ignore

    def predict_vo2(
        self,
        power: int | float | np.ndarray,
        weight: int | float,
        gender: Literal["Male", "Female"],
    ):
        """
        Predict oxygen consumption from cycling power output.

        Uses gender-specific ACSM equations to predict VO2 from power output
        normalized by body weight.

        Parameters
        ----------
        power : int, float, or np.ndarray
            Cycling power output in watts (W).
        weight : int or float
            Body weight in kilograms (kg).
        gender : {'Male', 'Female'}
            Participant gender to select appropriate equation.

        Returns
        -------
        vo2 : np.ndarray
            Predicted oxygen consumption in ml/kg/min.

        Raises
        ------
        ValueError
            If weight is not int or float, or if gender is not 'Male' or 'Female'.
        """
        if isinstance(power, (int, float)):
            power = np.array([power])
        power = power.astype(float).flatten()  # type: ignore
        if not isinstance(weight, (int, float)):
            raise ValueError("weight must be int or float.")
        if gender == "Male":
            eq = self.males_eq
        elif gender == "Female":
            eq = self.females_eq
        else:
            raise ValueError("gender must be Male or Female")
        eq = eq.subs(self.WEIGHT, weight)

        vo2 = [sympy.solve(eq.subs(self.POWER, p), self.VO2)[0] for p in power]
        return np.asarray(vo2, float)

    def predict_power(
        self,
        vo2: int | float | np.ndarray,
        weight: int | float,
        gender: Literal["Male", "Female"],
    ):
        """
        Predict cycling power output from oxygen consumption.

        Uses gender-specific ACSM equations to predict power output from VO2
        and body weight.

        Parameters
        ----------
        vo2 : int, float, or np.ndarray
            Oxygen consumption in ml/kg/min.
        weight : int or float
            Body weight in kilograms (kg).
        gender : {'Male', 'Female'}
            Participant gender to select appropriate equation.

        Returns
        -------
        power : np.ndarray
            Predicted cycling power output in watts (W).

        Raises
        ------
        ValueError
            If weight is not int or float, or if gender is not 'Male' or 'Female'.
        """
        if isinstance(vo2, (int, float)):
            vo2 = np.array([vo2])
        vo2 = vo2.astype(float).flatten()  # type: ignore
        if not isinstance(weight, (int, float)):
            raise ValueError("weight must be int or float.")
        if gender == "Male":
            eq = self.males_eq
        elif gender == "Female":
            eq = self.females_eq
        else:
            raise ValueError("gender must be Male or Female")
        eq = eq.subs(self.WEIGHT, weight)

        power = [sympy.solve(eq.subs(self.VO2, v), self.POWER)[0] for v in vo2]
        return np.asarray(power, float)
