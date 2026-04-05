"""strength module containing 1RM predicting equations"""

__all__ = ["Brzycki1RM"]


class Brzycki1RM:
    """
    Brzycki equation for 1-repetition maximum (1RM) prediction.

    This class implements the Brzycki equation for predicting one-repetition
    maximum (1RM) from submaximal loads and repetitions. The equation is valid
    for 1-36 repetitions.

    The Brzycki equation: 1RM = load × 36 / (37 - reps)

    Notes
    -----
    Valid range: 1-36 repetitions. Accuracy decreases with higher repetition counts.
    """

    def _validate_reps(self, reps: object):
        """private method used to validate the reps input"""
        if not isinstance(reps, int):
            raise ValueError("reps must be an int.")
        if reps < 0 or reps >= 37:
            raise ValueError("reps must be an int within the [1, 36] range.")

    def _validate_load(self, load: object):
        """private method used to validate the load input"""
        if not isinstance(load, (float, int)):
            raise ValueError("load must be a float or int.")
        if load < 0:
            raise ValueError("load must be >= 0")

    def _validate_1rm(self, rm1: object):
        """private method used to validate the 1rm input"""
        if not isinstance(rm1, (float, int)):
            raise ValueError("rm1 must be a float or int.")
        if rm1 < 0:
            raise ValueError("rm1 must be >= 0")

    def predict_1rm(self, reps: int, load: float | int):
        """
        Predict one-repetition maximum from submaximal load and repetitions.

        Calculate the estimated 1RM using the Brzycki equation from a submaximal
        load and the number of repetitions performed to failure.

        Parameters
        ----------
        reps : int
            Number of repetitions performed to failure (must be 1-36).
        load : float or int
            Submaximal load in kg used for the repetitions.

        Returns
        -------
        rm1 : float
            Predicted one-repetition maximum in kg.

        Raises
        ------
        ValueError
            If reps is outside the valid range [1, 36], or if load < 0.
        """
        self._validate_reps(reps)
        self._validate_load(load)
        return load * 36 / (37 - reps)

    def predict_reps(self, rm1: float | int, load: float | int):
        """
        Predict maximum repetitions from 1RM and load.

        Calculate the maximum number of repetitions that can be performed at a
        given load using the Brzycki equation rearranged for repetitions.

        Parameters
        ----------
        rm1 : float or int
            One-repetition maximum in kg.
        load : float or int
            Training load in kg (must be ≤ 1RM).

        Returns
        -------
        reps : float
            Predicted maximum number of repetitions at the given load.

        Raises
        ------
        ValueError
            If load > rm1, or if rm1 or load are invalid values.
        """
        self._validate_1rm(rm1)
        self._validate_load(load)
        if load > rm1:
            raise ValueError("load must be <= rm1.")
        return 37 - 36 * load / rm1

    def predict_load(self, rm1: float | int, reps: int):
        """
        Predict training load from 1RM and target repetitions.

        Calculate the load that allows a specific number of repetitions to be
        performed using the Brzycki equation rearranged for load.

        Parameters
        ----------
        rm1 : float or int
            One-repetition maximum in kg.
        reps : int
            Target number of repetitions (must be 1-36).

        Returns
        -------
        load : float
            Predicted training load in kg that allows the target repetitions.

        Raises
        ------
        ValueError
            If reps is outside the valid range [1, 36], or if rm1 is invalid.
        """
        self._validate_reps(reps)
        self._validate_1rm(rm1)
        return rm1 * (37 - reps) / 36
