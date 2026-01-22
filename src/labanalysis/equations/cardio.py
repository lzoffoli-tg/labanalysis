"""metabolic module containing VO2 predicting equations"""

#! IMPORTS

from typing import Literal
import sympy
import numpy as np

__all__ = ["Run", "Bike"]


class Run:

    GRADE = sympy.Symbol("GRADE", real=True)
    SPEED = sympy.Symbol("SPEED", real=True)
    VO2 = sympy.Symbol("VO2", real=True)
    VO2_WALKING = 3.5 + 5 / 3 * SPEED + 3 / 10 * SPEED * GRADE  # type: ignore
    VO2_RUNNING = 3.5 + 10 / 3 * SPEED + 3 / 20 * SPEED * GRADE  # type: ignore
    VO2_5kmh = VO2_WALKING.subs(SPEED, 5)
    VO2_7kmh = VO2_RUNNING.subs(SPEED, 7)
    VO2_transition = VO2_5kmh + (VO2_7kmh - VO2_5kmh) / 2 * (SPEED - sympy.Integer(5))
    walking = sympy.Equality(VO2, VO2_WALKING)
    running = sympy.Equality(VO2, VO2_RUNNING)
    transition = sympy.Equality(VO2, VO2_transition)

    def predict_vo2(
        self, speed: int | float | np.ndarray, grade: int | float | np.ndarray
    ):
        if isinstance(speed, (int, float)):
            speed = np.array([speed])
        speed = speed.astype(float).flatten()  # type: ignore
        if isinstance(grade, (int, float)):
            grade = np.array([grade])
        grade = grade.astype(float).flatten()  # type: ignore

        vo2 = []
        for s, g in zip(speed, grade):
            if s <= 5:
                eq = self.walking
            elif s >= 7:
                eq = self.running
            else:
                eq = self.transition
            vo2.append(
                sympy.solve(eq.subs(self.SPEED, s).subs(self.GRADE, g), self.VO2)[0]
            )

        return np.asarray(vo2, float)

    def predict_speed(
        self, vo2: int | float | np.ndarray, grade: int | float | np.ndarray
    ):
        if isinstance(vo2, (int, float)):
            vo2 = np.array([vo2])
        vo2 = vo2.astype(float).flatten()  # type: ignore
        if isinstance(grade, (int, float)):
            grade = np.array([grade])
        grade = grade.astype(float).flatten()  # type: ignore

        speed = []
        for v, g in zip(vo2, grade):
            v5 = self.VO2_5kmh.subs(self.GRADE, g)
            v7 = self.VO2_7kmh.subs(self.GRADE, g)
            if v <= v5:
                eq = self.walking.subs(self.VO2, v).subs(self.GRADE, g)
            elif v >= v7:
                eq = self.transition.subs(self.VO2, v).subs(self.GRADE, g)
            else:
                eq = self.running.subs(self.VO2, v).subs(self.GRADE, g)
            speed.append(sympy.solve(eq, self.SPEED)[0])

        return np.asarray(speed, float)

    def predict_grade(
        self, vo2: int | float | np.ndarray, speed: int | float | np.ndarray
    ):
        if isinstance(vo2, (int, float)):
            vo2 = np.array([vo2])
        vo2 = vo2.astype(float).flatten()  # type: ignore
        if isinstance(speed, (int, float)):
            speed = np.array([speed])
        speed = speed.astype(float).flatten()  # type: ignore

        grade = []
        for v, s in zip(vo2, speed):
            if s <= 5:
                eq = self.walking.subs(self.VO2, v).subs(self.SPEED, s)
            elif s >= 7:
                eq = self.running.subs(self.VO2, v).subs(self.SPEED, s)
            else:
                eq = self.transition.subs(self.VO2, v).subs(self.SPEED, s)
            grade.append(sympy.solve(eq, self.GRADE)[0])
        return np.asarray(grade, float)


class Bike:

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
        if isinstance(power, (int, float)):
            power = np.array([power])
        power = power.astype(float).flatten()  # type: ignore
        if not isinstance(weight, (int, float)):
            raise ValueError("weight must be int or float.")
        if gender == "Male":
            eq = self.males_eq.subs(self.WEIGHT, weight)
        elif gender == "Female":
            eq = self.females_eq.subs(self.WEIGHT, weight)
        else:
            raise ValueError("gender must be Male or Female")

        vo2 = [sympy.solve(eq.subs(self.POWER, p), self.VO2)[0] for p in power]
        return np.asarray(vo2, float)

    def predict_power(
        self,
        vo2: int | float | np.ndarray,
        weight: int | float,
        gender: Literal["Male", "Female"],
    ):
        if isinstance(vo2, (int, float)):
            vo2 = np.array([vo2])
        vo2 = vo2.astype(float).flatten()  # type: ignore
        if not isinstance(weight, (int, float)):
            raise ValueError("weight must be int or float.")
        if gender == "Male":
            eq = self.males_eq.subs(self.WEIGHT, weight)
        elif gender == "Female":
            eq = self.females_eq.subs(self.WEIGHT, weight)
        else:
            raise ValueError("gender must be Male or Female")

        power = [sympy.solve(eq.subs(self.VO2, v), self.POWER)[0] for v in vo2]
        return np.asarray(power, float)
