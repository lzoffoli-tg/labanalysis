"""ACSM running/walking metabolic equations for VO2 prediction."""

import numpy as np
import sympy


class Run:
    """
    ACSM running/walking metabolic equations for VO2 prediction.

    This class implements the American College of Sports Medicine (ACSM) metabolic
    equations for predicting oxygen consumption during walking and running on a treadmill.
    The equations automatically transition between walking (≤5 km/h), running (≥7 km/h),
    and an interpolated transition zone (5-7 km/h).

    Attributes
    ----------
    GRADE : sympy.Symbol
        Treadmill grade (%) symbolic variable.
    SPEED : sympy.Symbol
        Treadmill speed (km/h) symbolic variable.
    VO2 : sympy.Symbol
        Oxygen consumption (ml/kg/min) symbolic variable.
    VO2_WALKING : sympy.Expr
        ACSM walking equation: VO2 = 3.5 + 5/3*speed + 3/10*speed*grade
    VO2_RUNNING : sympy.Expr
        ACSM running equation: VO2 = 3.5 + 10/3*speed + 3/20*speed*grade
    walking : sympy.Equality
        Symbolic equality for walking equation.
    running : sympy.Equality
        Symbolic equality for running equation.
    transition : sympy.Equality
        Symbolic equality for transition zone (5-7 km/h).
    """

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
        """
        Predict oxygen consumption from treadmill speed and grade.

        Uses ACSM walking equation for speeds ≤5 km/h, running equation for speeds
        ≥7 km/h, and linear interpolation for the transition zone (5-7 km/h).

        Parameters
        ----------
        speed : int, float, or np.ndarray
            Treadmill speed in km/h.
        grade : int, float, or np.ndarray
            Treadmill grade in percent (%).

        Returns
        -------
        vo2 : np.ndarray
            Predicted oxygen consumption in ml/kg/min.
        """
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
        """
        Predict treadmill speed from oxygen consumption and grade.

        Automatically selects the appropriate ACSM equation (walking, running, or
        transition) based on the VO2 value relative to the thresholds at 5 and 7 km/h.

        Parameters
        ----------
        vo2 : int, float, or np.ndarray
            Oxygen consumption in ml/kg/min.
        grade : int, float, or np.ndarray
            Treadmill grade in percent (%).

        Returns
        -------
        speed : np.ndarray
            Predicted treadmill speed in km/h.
        """
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
            elif v < v7:
                eq = self.transition.subs(self.VO2, v).subs(self.GRADE, g)
            else:
                eq = self.running.subs(self.VO2, v).subs(self.GRADE, g)
            speed.append(sympy.solve(eq, self.SPEED)[0])

        return np.asarray(speed, float)

    def predict_grade(
        self, vo2: int | float | np.ndarray, speed: int | float | np.ndarray
    ):
        """
        Predict treadmill grade from oxygen consumption and speed.

        Uses ACSM walking equation for speeds ≤5 km/h, running equation for speeds
        ≥7 km/h, and linear interpolation for the transition zone (5-7 km/h).

        Parameters
        ----------
        vo2 : int, float, or np.ndarray
            Oxygen consumption in ml/kg/min.
        speed : int, float, or np.ndarray
            Treadmill speed in km/h.

        Returns
        -------
        grade : np.ndarray
            Predicted treadmill grade in percent (%).
        """
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
