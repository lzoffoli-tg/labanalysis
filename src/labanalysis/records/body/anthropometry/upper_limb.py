"""Auto-generated mixin for WholeBody properties."""

import warnings
import numpy as np
from ....timeseries import Signal1D

class UpperLimbMeasuresMixin:
    """UpperLimbMeasures properties for WholeBody."""

    @property
    def left_arm_length(self):
        """
        Calculate left arm length as distance from shoulder to elbow.
        Returns
        -------
        Signal1D
            Distance in meters from shoulder to elbow joint center.
        """
        try:
            shoulder = self.left_shoulder
            elbow = self.left_elbow
            index = np.unique(np.concatenate([shoulder.index, elbow.index])).tolist()
            data = np.asarray(elbow - shoulder)
            data = np.sum(data**2, axis=1) ** 0.5
            return Signal1D(data=data, index=index, unit=shoulder.unit)
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate left_arm_length: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def left_elbow_width(self):
        """
        Calculate left elbow width as distance between medial and lateral elbow markers.
        Returns
        -------
        Signal1D
            Distance in meters between medial and lateral elbow epicondyles.
        """
        medial = self._get_point("left_elbow_medial")
        lateral = self._get_point("left_elbow_lateral")
        if medial is None or lateral is None:
            warnings.warn(
                "Cannot calculate left_elbow_width: missing markers ['left_elbow_medial' or 'left_elbow_lateral']. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
        index = np.unique(np.concatenate([medial.index, lateral.index])).tolist()
        data = np.asarray(lateral - medial)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=medial.unit)

    @property
    def left_forearm_length(self):
        """
        Calculate left forearm length as distance from elbow to wrist.
        Returns
        -------
        Signal1D
            Distance in meters from elbow to wrist joint center.
        """
        try:
            elbow = self.left_elbow
            wrist = self.left_wrist
            index = np.unique(np.concatenate([elbow.index, wrist.index])).tolist()
            data = np.asarray(wrist - elbow)
            data = np.sum(data**2, axis=1) ** 0.5
            return Signal1D(data=data, index=index, unit=elbow.unit)
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate left_forearm_length: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def left_upper_limb_length(self):
        """
        Calculate total left upper limb length (arm + forearm).
        Returns the sum of arm length (shoulder to elbow) and forearm length
        (elbow to wrist). This represents the functional length of the
        entire upper limb.
        Returns
        -------
        Signal1D
            Distance in meters from shoulder to wrist joint center.
        See Also
        --------
        left_arm_length : Length of arm segment only
        left_forearm_length : Length of forearm segment only
        right_upper_limb_length : Right upper limb total length
        """
        return self.left_arm_length + self.left_forearm_length

    @property
    def right_arm_length(self):
        """
        Calculate right arm length as distance from shoulder to elbow.
        Returns
        -------
        Signal1D
            Distance in meters from shoulder to elbow joint center.
        """
        try:
            shoulder = self.right_shoulder
            elbow = self.right_elbow
            index = np.unique(np.concatenate([shoulder.index, elbow.index])).tolist()
            data = np.asarray(elbow - shoulder)
            data = np.sum(data**2, axis=1) ** 0.5
            return Signal1D(data=data, index=index, unit=shoulder.unit)
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate right_arm_length: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def right_elbow_width(self):
        """
        Calculate right elbow width as distance between medial and lateral elbow markers.
        Returns
        -------
        Signal1D
            Distance in meters between medial and lateral elbow epicondyles.
        """
        medial = self._get_point("right_elbow_medial")
        lateral = self._get_point("right_elbow_lateral")
        if medial is None or lateral is None:
            warnings.warn(
                "Cannot calculate right_elbow_width: missing markers ['right_elbow_medial' or 'right_elbow_lateral']. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)
        index = np.unique(np.concatenate([medial.index, lateral.index])).tolist()
        data = np.asarray(lateral - medial)
        data = np.sum(data**2, axis=1) ** 0.5
        return Signal1D(data=data, index=index, unit=medial.unit)

    @property
    def right_forearm_length(self):
        """
        Calculate right forearm length as distance from elbow to wrist.
        Returns
        -------
        Signal1D
            Distance in meters from elbow to wrist joint center.
        """
        try:
            elbow = self.right_elbow
            wrist = self.right_wrist
            index = np.unique(np.concatenate([elbow.index, wrist.index])).tolist()
            data = np.asarray(wrist - elbow)
            data = np.sum(data**2, axis=1) ** 0.5
            return Signal1D(data=data, index=index, unit=elbow.unit)
        except (AttributeError, TypeError, ValueError):
            warnings.warn(
                "Cannot calculate right_forearm_length: missing required markers. Returning NaN.",
                UserWarning
            )
            ref = self._find_any_valid_marker()
            return self._create_nan_signal1d(ref)

    @property
    def right_upper_limb_length(self):
        """
        Calculate total right upper limb length (arm + forearm).
        Returns the sum of arm length (shoulder to elbow) and forearm length
        (elbow to wrist). This represents the functional length of the
        entire upper limb.
        Returns
        -------
        Signal1D
            Distance in meters from shoulder to wrist joint center.
        See Also
        --------
        right_arm_length : Length of arm segment only
        right_forearm_length : Length of forearm segment only
        left_upper_limb_length : Left upper limb total length
        """
        return self.right_arm_length + self.right_forearm_length
