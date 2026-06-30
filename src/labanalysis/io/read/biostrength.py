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

    def __copy__(self):
        """Support shallow copy of wrapper."""
        import copy

        new_wrapper = self.__class__.__new__(self.__class__)
        new_wrapper._product = copy.copy(self._product)
        return new_wrapper

    def __deepcopy__(self, memo):
        """Support deep copy of wrapper."""
        import copy

        new_wrapper = self.__class__.__new__(self.__class__)
        new_wrapper._product = copy.deepcopy(self._product, memo)
        return new_wrapper

    def __getitem__(self, key):
        """
        Support slicing and indexing.

        Returns a new instance with sliced data by accessing the external product's
        raw arrays and creating a new external instance.
        """
        # Get raw arrays from external product
        time_arr = np.array(self._product.time_s)
        motor_position_arr = np.array(self._product.motor_position_rad)
        motor_torque_arr = np.array(self._product.motor_torque_nm)

        # Apply slicing
        sliced_time = time_arr[key]
        sliced_position = motor_position_arr[key]
        sliced_torque = motor_torque_arr[key]

        # Ensure arrays (convert single values to lists)
        if np.isscalar(sliced_time):
            time_list = [float(sliced_time)]
            position_list = [float(sliced_position)]
            torque_list = [float(sliced_torque)]
        else:
            time_list = sliced_time.tolist()
            position_list = sliced_position.tolist()
            torque_list = sliced_torque.tolist()

        # Create new instance with sliced data
        new_instance = self.__class__.__new__(self.__class__)
        new_instance._product = self._external_class(
            time_s=time_list,
            motor_position_rad=position_list,
            motor_torque_nm=torque_list
        )
        return new_instance

    def __len__(self):
        """Return number of samples."""
        return len(self._product.time_s)

    def __init_subclass__(cls):
        """
        Automatically copy _rm1_coefs from external class when subclass is created.

        This ensures wrapper classes expose the same _rm1_coefs as the external class.
        """
        super().__init_subclass__()
        if cls._external_class is not None and hasattr(
            cls._external_class, "_rm1_coefs"
        ):
            cls._rm1_coefs = cls._external_class._rm1_coefs


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

    def __init__(
        self, time_s, motor_position_rad, motor_load_nm, roll_position: int = 18
    ):
        """
        Initialize LegExtensionREV with roll_position.

        Note: External package uses default roll_position=11, internal used 18.
        We keep 18 as default for backward compatibility.
        """
        # External package expects roller_position (default 11), we map our 18 default
        self._product = self._external_class(
            time_s,
            motor_position_rad,
            motor_load_nm,
            roller_position=roll_position,
        )

    @classmethod
    def from_txt_file(cls, file: str, roll_position: int = 18):
        """Read from file with roll_position parameter."""
        external_instance = cls._external_class.from_txt(
            file,
            roller_position=roll_position,
        )
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
