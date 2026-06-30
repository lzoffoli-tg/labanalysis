"""Biostrength data reader (wrapper around biostrengthdataconverter package)."""

import numpy as np
from biostrengthdataconverter import Biostrength as _Biostrength

G = 9.80665


class BiostrengthProduct:
    """Wrapper base class maintaining backward compatibility with external package."""

    _external_class = None  # Set in subclasses

    def __init__(self, time_s, motor_position_rad, motor_load_nm):
        """Initialize wrapper by delegating to external package class."""
        self._product = self._external_class(time_s, motor_position_rad, motor_load_nm)

    @classmethod
    def from_txt_file(cls, file: str):
        """Read data from file (maps to external package's from_txt method)."""
        external_instance = cls._external_class.from_txt(file)
        wrapper = cls.__new__(cls)
        wrapper._product = external_instance
        return wrapper

    @property
    def time_s(self):
        """Return time array (converts list to numpy array)."""
        return np.array(self._product.time_s)

    @property
    def load_kgf(self):
        """Return load in kgf (converts force_N to load_kgf)."""
        return np.array(self._product.force_N) / G

    @property
    def position_lever_m(self):
        """Return lever position in meters (converts displacement_m)."""
        return np.array(self._product.displacement_m)


# Product wrapper classes
class ChestPress(BiostrengthProduct):
    _external_class = _Biostrength.ChestPress


class ShoulderPress(BiostrengthProduct):
    _external_class = _Biostrength.ShoulderPress


class LowRow(BiostrengthProduct):
    _external_class = _Biostrength.LowRow


class LegPress(BiostrengthProduct):
    _external_class = _Biostrength.LegPress


class LegExtension(BiostrengthProduct):
    _external_class = _Biostrength.LegExtension


class LegCurl(BiostrengthProduct):
    _external_class = _Biostrength.LegCurl


class AdjustablePulleyREV(BiostrengthProduct):
    _external_class = _Biostrength.AdjustablePulleyREV


class LegPressREV(BiostrengthProduct):
    _external_class = _Biostrength.LegPressREV


class LegExtensionREV(BiostrengthProduct):
    """LegExtensionREV with roll_position parameter (maps to external package)."""

    _external_class = _Biostrength.LegExtensionREV

    def __init__(self, time_s, motor_position_rad, motor_load_nm, roll_position: int = 18):
        """
        Initialize LegExtensionREV with roll_position.

        Note: External package uses default roll_position=11, internal used 18.
        We keep 18 as default for backward compatibility.
        """
        # External package expects roll_position (default 11), we map our 18 default
        self._product = self._external_class(
            time_s, motor_position_rad, motor_load_nm, roll_position=roll_position
        )

    @classmethod
    def from_txt_file(cls, file: str, roll_position: int = 18):
        """Read from file with roll_position parameter."""
        external_instance = cls._external_class.from_txt(file, roll_position=roll_position)
        wrapper = cls.__new__(cls)
        wrapper._product = external_instance
        return wrapper


# PRODUCTS dictionary for compatibility
PRODUCTS = {
    "CHEST PRESS": ChestPress,
    "SHOULDER PRESS": ShoulderPress,
    "LOW ROW": LowRow,
    "LEG PRESS": LegPress,
    "LEG EXTENSION": LegExtension,
    "ADJUSTABLE PULLEY REV": AdjustablePulleyREV,
    "LEG PRESS REV": LegPressREV,
    "LEG EXTENSION REV": LegExtensionREV,
    "LEG CURL": LegCurl,
}
