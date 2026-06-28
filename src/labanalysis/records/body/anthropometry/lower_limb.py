"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Signal1D

class LowerLimbMeasuresMixin:
    """LowerLimbMeasures properties for WholeBody."""

    @property
    def left_ankle_width(self):
        """
        Calculate left ankle width as distance between medial and lateral ankle markers.
        Returns
        -------
        Signal1D
            Distance in meters between medial and lateral ankle malleoli.
        """
        medial = self._get_point("left_ankle_medial")
        lateral = self._get_point("left_ankle_lateral")
        if medial is None or lateral is None:
            warnings.warn(
                "Cannot calculate left_ankle_width: missing markers ['left_ankle_medial' or 'left_ankle_lateral']. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
        index = np.unique(np.concatenate([medial.index, lateral.index])).tolist()
        data = np.asarray(lateral - medial)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=medial.unit)

    @property
    def left_knee_width(self):
        """
        Calculate left knee width as distance between medial and lateral knee markers.
        Returns
        -------
        Signal1D
            Distance in meters between medial and lateral femoral epicondyles.
        """
        medial = self._get_point("left_knee_medial")
        lateral = self._get_point("left_knee_lateral")
        if medial is None or lateral is None:
            warnings.warn(
                "Cannot calculate left_knee_width: missing markers ['left_knee_medial' or 'left_knee_lateral']. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
        index = np.unique(np.concatenate([medial.index, lateral.index])).tolist()
        data = np.asarray(lateral - medial)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=medial.unit)

    @property
    def left_leg_length(self):
        """
        Calculate left leg length as distance from ankle to knee.
        Returns
        -------
        Signal1D
            Distance in meters from ankle to knee joint center.
        """
        try:
            ankle = self.left_ankle
            knee = self.left_knee
            index = np.unique(np.concatenate([ankle.index, knee.index])).tolist()
            data = np.asarray(knee - ankle)
            data = np.sum(data**2, axis=1) ** 0.5
            return Signal1D(data=data, index=index, unit=ankle.unit)
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate left_leg_length: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def left_lower_limb_length(self):
        """
        Calculate total left lower limb length (thigh + leg).
        Returns the sum of thigh length (hip to knee) and leg length
        (knee to ankle). This represents the functional length of the
        entire lower limb.
        Returns
        -------
        Signal1D
            Distance in meters from hip to ankle joint center.
        See Also
        --------
        left_thigh_length : Length of thigh segment only
        left_leg_length : Length of leg segment only
        right_lower_limb_length : Right lower limb total length
        """
        return self.left_thigh_length + self.left_leg_length

    @property
    def left_thigh_length(self):
        """
        Calculate left thigh length as distance from knee to hip.
        Returns
        -------
        Signal1D
            Distance in meters from knee to hip joint center.
        """
        try:
            knee = self.left_knee
            hip = self.left_hip
            index = np.unique(np.concatenate([knee.index, hip.index])).tolist()
            data = np.asarray(hip - knee)
            data = np.sum(data**2, axis=1) ** 0.5
            return Signal1D(data=data, index=index, unit=knee.unit)
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate left_thigh_length: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def right_ankle_width(self):
        """
        Calculate right ankle width as distance between medial and lateral ankle markers.
        Returns
        -------
        Signal1D
            Distance in meters between medial and lateral ankle malleoli.
        """
        medial = self._get_point("right_ankle_medial")
        lateral = self._get_point("right_ankle_lateral")
        if medial is None or lateral is None:
            warnings.warn(
                "Cannot calculate right_ankle_width: missing markers ['right_ankle_medial' or 'right_ankle_lateral']. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
        index = np.unique(np.concatenate([medial.index, lateral.index])).tolist()
        data = np.asarray(lateral - medial)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=medial.unit)

    @property
    def right_knee_width(self):
        """
        Calculate right knee width as distance between medial and lateral knee markers.
        Returns
        -------
        Signal1D
            Distance in meters between medial and lateral femoral epicondyles.
        """
        medial = self._get_point("right_knee_medial")
        lateral = self._get_point("right_knee_lateral")
        if medial is None or lateral is None:
            warnings.warn(
                "Cannot calculate right_knee_width: missing markers ['right_knee_medial' or 'right_knee_lateral']. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
        index = np.unique(np.concatenate([medial.index, lateral.index])).tolist()
        data = np.asarray(lateral - medial)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=medial.unit)

    @property
    def right_leg_length(self):
        """
        Calculate right leg length as distance from ankle to knee.
        Returns
        -------
        Signal1D
            Distance in meters from ankle to knee joint center.
        """
        try:
            ankle = self.right_ankle
            knee = self.right_knee
            index = np.unique(np.concatenate([ankle.index, knee.index])).tolist()
            data = np.asarray(knee - ankle)
            data = np.sum(data**2, axis=1) ** 0.5
            return Signal1D(data=data, index=index, unit=ankle.unit)
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate right_leg_length: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def right_lower_limb_length(self):
        """
        Calculate total right lower limb length (thigh + leg).
        Returns the sum of thigh length (hip to knee) and leg length
        (knee to ankle). This represents the functional length of the
        entire lower limb.
        Returns
        -------
        Signal1D
            Distance in meters from hip to ankle joint center.
        See Also
        --------
        right_thigh_length : Length of thigh segment only
        right_leg_length : Length of leg segment only
        left_lower_limb_length : Left lower limb total length
        """
        return self.right_thigh_length + self.right_leg_length

    @property
    def right_thigh_length(self):
        """
        Calculate right thigh length as distance from knee to hip.
        Returns
        -------
        Signal1D
            Distance in meters from knee to hip joint center.
        """
        try:
            knee = self.right_knee
            hip = self.right_hip
            index = np.unique(np.concatenate([knee.index, hip.index])).tolist()
            data = np.asarray(hip - knee)
            data = np.sum(data**2, axis=1) ** 0.5
            return Signal1D(data=data, index=index, unit=knee.unit)
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate right_thigh_length: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
