"""
Test suite for labanalysis.io.read.biostrength module.

Tests verify BiostrengthProduct class and related file reading functions.
"""

import pytest

from labanalysis.io.read import biostrength


def test_biostrength_module_importable():
    """
    Test that biostrength module imports without errors.

    Expected:
        Module should be importable
    """
    assert biostrength is not None


def test_biostrengthproduct_class_exists():
    """
    Test that BiostrengthProduct class is defined.

    Expected:
        Class should exist and be instantiable
    """
    assert hasattr(biostrength, 'BiostrengthProduct')
    assert isinstance(biostrength.BiostrengthProduct, type)


def test_biostrengthproduct_class_attributes():
    """
    Test that BiostrengthProduct wrapper has expected structure.

    Expected:
        Wrapper class should have _external_class attribute pointing to
        biostrengthdataconverter package classes.
    """
    cls = biostrength.BiostrengthProduct
    # Wrapper must have _external_class attribute (set in subclasses)
    assert hasattr(cls, '_external_class'), "Missing _external_class attribute"

    # Verify subclasses have _external_class set
    from biostrengthdataconverter import Biostrength as _Biostrength
    assert biostrength.ChestPress._external_class == _Biostrength.ChestPress
    assert biostrength.LegPress._external_class == _Biostrength.LegPress


def test_biostrengthproduct_roller_position_parameter():
    """
    Test that wrapper correctly maps roll_position to roller_position parameter.

    This test ensures that the parameter name mismatch bug doesn't reoccur.
    The external biostrengthdataconverter package uses 'roller_position' for
    LegExtensionREV, while our wrapper API uses 'roll_position' for backward compatibility.

    Expected:
        - LegExtensionREV constructor should accept roll_position and pass it as roller_position
        - LegExtensionREV.from_txt_file should accept roll_position and pass it as roller_position
    """
    import inspect
    from unittest.mock import Mock, patch

    # LegExtensionREV is the only product that uses roller_position parameter
    product_class = biostrength.LegExtensionREV
    external_class = product_class._external_class

    # Verify external class has 'roller_position' parameter (not 'roll_position')
    init_sig = inspect.signature(external_class.__init__)
    assert 'roller_position' in init_sig.parameters, (
        f"{external_class.__name__}.__init__ missing 'roller_position' parameter"
    )
    assert 'roll_position' not in init_sig.parameters, (
        f"{external_class.__name__}.__init__ should not have 'roll_position' parameter"
    )

    from_txt_sig = inspect.signature(external_class.from_txt)
    assert 'roller_position' in from_txt_sig.parameters, (
        f"{external_class.__name__}.from_txt missing 'roller_position' parameter"
    )
    assert 'roll_position' not in from_txt_sig.parameters, (
        f"{external_class.__name__}.from_txt should not have 'roll_position' parameter"
    )

    # Test that wrapper correctly passes the parameter
    with patch.object(biostrength.LegExtensionREV, '_external_class') as mock_external:
        mock_instance = Mock()
        mock_external.return_value = mock_instance
        mock_external.from_txt.return_value = mock_instance

        # Test constructor call
        try:
            product = biostrength.LegExtensionREV(
                time_s=[0.0, 1.0],
                motor_position_rad=[0.0, 1.0],
                motor_load_nm=[0.0, 10.0],
                roll_position=18
            )
            # Verify it was called with roller_position, not roll_position
            mock_external.assert_called_once()
            call_kwargs = mock_external.call_args.kwargs
            assert 'roller_position' in call_kwargs, "Constructor should use 'roller_position'"
            assert call_kwargs['roller_position'] == 18
            assert 'roll_position' not in call_kwargs, "Constructor should not use 'roll_position'"
        except TypeError as e:
            if "roll_position" in str(e):
                pytest.fail(f"Constructor incorrectly uses 'roll_position' instead of 'roller_position': {e}")
            raise

        # Reset mock for next test
        mock_external.reset_mock()

        # Test from_txt_file call
        try:
            product = biostrength.LegExtensionREV.from_txt_file(
                "dummy_file.txt",
                roll_position=18
            )
            # Verify it was called with roller_position, not roll_position
            mock_external.from_txt.assert_called_once()
            call_kwargs = mock_external.from_txt.call_args.kwargs
            assert 'roller_position' in call_kwargs, "from_txt should use 'roller_position'"
            assert call_kwargs['roller_position'] == 18
            assert 'roll_position' not in call_kwargs, "from_txt should not use 'roll_position'"
        except TypeError as e:
            if "roll_position" in str(e):
                pytest.fail(f"from_txt_file incorrectly uses 'roll_position' instead of 'roller_position': {e}")
            raise


def test_biostrengthproduct_rm1_coefs_attribute():
    """
    Test that wrapper classes expose _rm1_coefs attribute from external classes.

    This test ensures that the _rm1_coefs attribute is accessible on wrapper classes,
    which is required by isokinetic_1rm_test.py and other code that needs 1RM coefficients.

    Expected:
        - All product classes should have _rm1_coefs attribute
        - Values should match those from the external biostrengthdataconverter package
    """
    from biostrengthdataconverter import Biostrength as _Biostrength

    # Test all product classes
    product_classes = [
        (biostrength.ChestPress, _Biostrength.ChestPress),
        (biostrength.ShoulderPress, _Biostrength.ShoulderPress),
        (biostrength.LowRow, _Biostrength.LowRow),
        (biostrength.LegPress, _Biostrength.LegPress),
        (biostrength.LegExtension, _Biostrength.LegExtension),
        (biostrength.LegCurl, _Biostrength.LegCurl),
        (biostrength.AdjustablePulleyREV, _Biostrength.AdjustablePulleyREV),
        (biostrength.LegPressREV, _Biostrength.LegPressREV),
        (biostrength.LegExtensionREV, _Biostrength.LegExtensionREV),
    ]

    for wrapper_class, external_class in product_classes:
        # Verify wrapper has _rm1_coefs
        assert hasattr(wrapper_class, '_rm1_coefs'), (
            f"{wrapper_class.__name__} missing _rm1_coefs attribute"
        )

        # Verify external class has _rm1_coefs
        assert hasattr(external_class, '_rm1_coefs'), (
            f"External {external_class.__name__} missing _rm1_coefs attribute"
        )

        # Verify values match
        assert wrapper_class._rm1_coefs == external_class._rm1_coefs, (
            f"{wrapper_class.__name__}._rm1_coefs ({wrapper_class._rm1_coefs}) "
            f"does not match external class ({external_class._rm1_coefs})"
        )

        # Verify it's a tuple of two floats (beta1, beta0)
        assert isinstance(wrapper_class._rm1_coefs, tuple), (
            f"{wrapper_class.__name__}._rm1_coefs should be a tuple"
        )
        assert len(wrapper_class._rm1_coefs) == 2, (
            f"{wrapper_class.__name__}._rm1_coefs should have 2 elements (beta1, beta0)"
        )


def _generate_synthetic_biostrength_txt(
    filepath: str,
    n_repetitions: int = 3,
    rep_duration_s: float = 6.5,
    sample_rate_hz: float = 50.0,
    torque_peak_nm: float = 35.0,
    position_range_rad: tuple = (1.1, 7.5),
    roller_position: int = 11,
):
    """
    Generate a synthetic Biostrength .txt file with realistic exercise patterns.

    Mimics real isokinetic/isometric exercise data with:
    - Cyclic repetitions (concentric + eccentric phases)
    - Realistic torque-position relationship
    - Smooth transitions and physiological patterns

    File format matches real Biostrength device output:
    - Header row with column names
    - Data rows with pipe-separated values
    - Time in format XXXX.XXX (milliseconds with 3 decimals)
    - Comma as decimal separator (European format)

    Args:
        filepath: Path where to save the synthetic file
        n_repetitions: Number of exercise repetitions
        rep_duration_s: Duration of each repetition (concentric + eccentric)
        sample_rate_hz: Sampling rate in Hz
        torque_peak_nm: Peak torque during maximal effort
        position_range_rad: Min and max joint position in radians
        roller_position: Roller position setting
    """
    import numpy as np

    # Calculate total duration and samples
    duration_s = n_repetitions * rep_duration_s
    n_samples = int(duration_s * sample_rate_hz)
    time_s = np.linspace(0, duration_s, n_samples)

    # Initialize arrays
    position_rad = np.zeros(n_samples)
    torque_nm = np.zeros(n_samples)

    # Generate each repetition
    samples_per_rep = int(rep_duration_s * sample_rate_hz)
    for rep in range(n_repetitions):
        start_idx = rep * samples_per_rep
        end_idx = min(start_idx + samples_per_rep, n_samples)
        n_rep_samples = end_idx - start_idx

        # Time within this repetition
        t_rep = np.linspace(0, 1, n_rep_samples)

        # Position: smooth sinusoidal movement (concentric -> eccentric)
        # Start at rest position, move to flexed, return to rest
        pos_min, pos_max = position_range_rad
        position_rad[start_idx:end_idx] = pos_min + (pos_max - pos_min) * (
            0.5 - 0.5 * np.cos(2 * np.pi * t_rep)
        )

        # Torque: follows position with lag, peaks during concentric phase
        # Realistic torque curve: bell-shaped during effort
        torque_profile = torque_peak_nm * np.sin(np.pi * t_rep) ** 2
        # Add variability (fatigue effect: slight decrease across reps)
        fatigue_factor = 1.0 - 0.1 * (rep / max(1, n_repetitions - 1))
        torque_nm[start_idx:end_idx] = torque_profile * fatigue_factor

        # Add measurement noise
        noise = np.random.normal(0, torque_peak_nm * 0.02, n_rep_samples)
        torque_nm[start_idx:end_idx] += noise

    # Ensure non-negative torque (can't have negative muscle force)
    torque_nm = np.maximum(torque_nm, 0)

    # Velocity (derivative of position)
    velocity_rad = np.gradient(position_rad, time_s)

    # Position in mm (lever arm conversion, approximate)
    position_mm = position_rad * 100

    # Write file in Biostrength format
    with open(filepath, 'w') as f:
        # Header
        f.write("0000.035|RX|UserTorque(Nm)|Velocity(rad)|Position(mm)|Position(rad)|Roller|Cell1|Cell2|TX|CmdTorque|EccOverload|EasyStart|Inertia|Friction|Vel_H|Vel_L|ViscoConc|ViscoEcc\n")

        # Data rows
        for i in range(n_samples):
            time_str = f"{time_s[i]:08.3f}".replace('.', ',')
            torque_str = f"{torque_nm[i]:.2f}".replace('.', ',')
            velocity_str = f"{velocity_rad[i]:.2f}".replace('.', ',')
            pos_mm_str = f"{position_mm[i]:.2f}".replace('.', ',')
            pos_rad_str = f"{position_rad[i]:.2f}".replace('.', ',')

            # Format row (simplified, only essential columns)
            row = f"{time_str}|RX|{torque_str}|{velocity_str}|{pos_mm_str}|{pos_rad_str}|{roller_position}|0,00|0,00|TX|-15,62|0,00|1,11|0,70|0,20|20,00|-20,00|0,00|0,00\n"
            f.write(row)


def test_biostrengthproduct_from_synthetic_file():
    """
    Integration test using synthetically generated Biostrength data.

    This test verifies the complete workflow:
        - Generate synthetic data file
        - Read with wrapper classes
        - Verify data properties are accessible
        - Verify _rm1_coefs is accessible
        - Test with different product types
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test LegCurl with realistic isokinetic exercise parameters
        legcurl_path = Path(tmpdir) / "legcurl_synthetic.txt"
        _generate_synthetic_biostrength_txt(
            str(legcurl_path),
            n_repetitions=3,
            rep_duration_s=6.5,
            torque_peak_nm=35.0,
            position_range_rad=(1.1, 7.5),
            roller_position=11
        )

        product = biostrength.LegCurl.from_txt_file(str(legcurl_path))

        # Verify data is loaded
        assert product.time_s is not None
        assert len(product.time_s) > 0
        assert product.load_kgf is not None
        assert len(product.load_kgf) > 0
        assert product.position_lever_m is not None
        assert len(product.position_lever_m) > 0

        # Verify arrays have same length
        assert len(product.time_s) == len(product.load_kgf)
        assert len(product.time_s) == len(product.position_lever_m)

        # Verify _rm1_coefs is accessible (needed for 1RM calculations)
        assert hasattr(biostrength.LegCurl, '_rm1_coefs')
        assert isinstance(biostrength.LegCurl._rm1_coefs, tuple)
        assert len(biostrength.LegCurl._rm1_coefs) == 2

        # Test LegExtensionREV with roller_position parameter and isometric pattern
        legext_path = Path(tmpdir) / "legextensionrev_synthetic.txt"
        _generate_synthetic_biostrength_txt(
            str(legext_path),
            n_repetitions=5,
            rep_duration_s=4.0,
            torque_peak_nm=50.0,
            position_range_rad=(0.5, 2.5),
            roller_position=18
        )

        # Test with default roll_position
        product_default = biostrength.LegExtensionREV.from_txt_file(str(legext_path))
        assert product_default.time_s is not None
        assert len(product_default.time_s) > 0

        # Test with custom roll_position
        product_custom = biostrength.LegExtensionREV.from_txt_file(str(legext_path), roll_position=11)
        assert product_custom.time_s is not None
        assert len(product_custom.time_s) > 0

        # Verify _rm1_coefs is accessible
        assert hasattr(biostrength.LegExtensionREV, '_rm1_coefs')
        assert isinstance(biostrength.LegExtensionREV._rm1_coefs, tuple)
        assert len(biostrength.LegExtensionREV._rm1_coefs) == 2


def test_biostrengthproduct_copy_operations():
    """
    Test that BiostrengthProduct wrapper supports copy operations.

    This verifies that:
        - Shallow copy works
        - Deep copy works
        - Copied instances are independent
    """
    import copy
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate synthetic data
        filepath = Path(tmpdir) / "test_copy.txt"
        _generate_synthetic_biostrength_txt(
            str(filepath),
            n_repetitions=2,
            rep_duration_s=5.0,
            torque_peak_nm=30.0
        )

        # Load product
        original = biostrength.LegCurl.from_txt_file(str(filepath))

        # Test shallow copy
        shallow = copy.copy(original)
        assert shallow is not original
        assert len(shallow.time_s) == len(original.time_s)
        assert shallow.time_s[0] == original.time_s[0]

        # Test deep copy
        deep = copy.deepcopy(original)
        assert deep is not original
        assert len(deep.time_s) == len(original.time_s)
        assert deep.time_s[0] == original.time_s[0]


def test_biostrengthproduct_slicing_operations():
    """
    Test that BiostrengthProduct wrapper supports slicing and indexing.

    This verifies that:
        - Indexing single elements works
        - Slicing ranges works
        - Sliced instances maintain data integrity
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate synthetic data
        filepath = Path(tmpdir) / "test_slice.txt"
        _generate_synthetic_biostrength_txt(
            str(filepath),
            n_repetitions=3,
            rep_duration_s=6.0,
            torque_peak_nm=35.0,
            sample_rate_hz=50.0
        )

        # Load product
        product = biostrength.LegCurl.from_txt_file(str(filepath))

        # Verify original length
        original_len = len(product)
        assert original_len > 100  # Should have plenty of samples

        # Test single index
        single = product[0]
        assert len(single) == 1
        assert single.time_s[0] == product.time_s[0]

        # Test slice
        sliced = product[10:20]
        assert len(sliced) == 10
        assert sliced.time_s[0] == product.time_s[10]
        assert sliced.time_s[-1] == product.time_s[19]

        # Test negative indexing
        last = product[-1]
        assert len(last) == 1
        assert last.time_s[0] == product.time_s[-1]

        # Test slice with step
        stepped = product[0:100:10]
        assert len(stepped) == 10
        assert stepped.time_s[0] == product.time_s[0]
        assert stepped.time_s[1] == product.time_s[10]

        # Verify all properties are accessible on sliced data
        assert sliced.load_kgf is not None
        assert sliced.position_lever_m is not None
        assert len(sliced.load_kgf) == 10
        assert len(sliced.position_lever_m) == 10
