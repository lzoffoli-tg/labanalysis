"""Test fillna functionality at all levels"""

import sys
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# add project root to path like other tests do
sys.path.append(dirname(dirname(abspath(__file__))))

import src.labanalysis as laban
from src.labanalysis.records.records import Record
from src.labanalysis.records.timeseries import EMGSignal, Point3D, Signal1D, Signal3D, Timeseries


def test_fillna_with_constant_value():
    """Test fillna with constant value on Timeseries"""
    print("\n=== Test fillna with constant value ===")

    # Create test data with NaNs
    data = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, np.nan], [5.0, 6.0]])
    index = np.array([0.0, 1.0, 2.0, 3.0])
    columns = ["col1", "col2"]

    ts = Timeseries(data, index, columns, "m")

    # Fill with constant value
    filled = ts.fillna(value=0.0)

    # Verify
    assert not np.any(np.isnan(filled.to_numpy())), "There should be no NaNs after fillna with value"
    # Direct numpy array access (row, col)
    assert filled.to_numpy()[1, 0] == 0.0, "NaN at row 1, col 0 should be replaced with 0.0"
    assert filled.to_numpy()[2, 1] == 0.0, "NaN at row 2, col 1 should be replaced with 0.0"
    print("✓ Constant value fillna works correctly")


def test_fillna_with_interpolation():
    """Test fillna with cubic spline interpolation on Signal1D"""
    print("\n=== Test fillna with interpolation ===")

    # Create test data with NaNs
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    index = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    sig = Signal1D(data, index, "V")

    # Fill with interpolation (no value, no regressors)
    filled = sig.fillna()

    # Verify
    assert not np.any(np.isnan(filled.to_numpy())), "There should be no NaNs after fillna with interpolation"
    # The interpolated value at index 2 should be approximately 3.0
    # Use ix indexer for positional access
    interpolated_value = filled.to_numpy()[2, 0]
    assert 2.5 <= interpolated_value <= 3.5, f"Interpolated value {interpolated_value} should be close to 3.0"
    print(f"✓ Interpolation fillna works correctly (interpolated value: {interpolated_value:.3f})")


def test_fillna_with_regressors():
    """Test fillna with regressors on Signal3D"""
    print("\n=== Test fillna with regressors ===")

    # Create test data with controlled relationship
    n_samples = 50
    index = np.linspace(0, 5, n_samples)

    # Create regressors (independent variables)
    reg1 = np.sin(index)
    reg2 = np.cos(index)
    regressors = np.column_stack([reg1, reg2])

    # Create dependent variable with known relationship: y = 2*reg1 + 3*reg2 + noise
    y_true = 2 * reg1 + 3 * reg2 + np.random.randn(n_samples) * 0.01

    # Introduce NaNs
    y_with_nan = y_true.copy()
    nan_indices = [10, 20, 30]
    y_with_nan[nan_indices] = np.nan

    # Create 3D signal (replicate y_true for 3 columns)
    data_3d = np.column_stack([y_with_nan, y_with_nan + 1, y_with_nan - 1])
    sig3d = Signal3D(data_3d, index, "m")

    # Fill with regressors
    filled = sig3d.fillna(regressors=regressors)

    # Verify
    assert not np.any(np.isnan(filled.to_numpy())), "There should be no NaNs after fillna with regressors"

    # Check that filled values are reasonable
    for idx in nan_indices:
        filled_value = filled.to_numpy()[idx, 0]
        true_value = y_true[idx]
        error = abs(filled_value - true_value)
        assert error < 1.0, f"Filled value {filled_value:.3f} should be close to true value {true_value:.3f}"

    print("✓ Regression-based fillna works correctly")


def test_fillna_on_point3d():
    """Test fillna on Point3D class"""
    print("\n=== Test fillna on Point3D ===")

    # Create test data
    data = np.array([
        [1.0, 2.0, 3.0],
        [np.nan, 2.1, 3.1],
        [1.2, np.nan, 3.2],
        [1.3, 2.3, np.nan],
        [1.4, 2.4, 3.4]
    ])
    index = np.linspace(0, 1, 5)

    point = Point3D(data, index, "m")

    # Fill with interpolation
    filled = point.fillna()

    # Verify
    assert not np.any(np.isnan(filled.to_numpy())), "There should be no NaNs after fillna"
    assert filled.unit == "m", "Unit should be preserved"
    print("✓ Point3D fillna works correctly")


def test_fillna_on_emgsignal():
    """Test fillna on EMGSignal class"""
    print("\n=== Test fillna on EMGSignal ===")

    # Create test data
    data = np.array([100.0, 150.0, np.nan, 200.0, 250.0])
    index = np.array([0.0, 0.01, 0.02, 0.03, 0.04])

    emg = EMGSignal(data, index, "biceps", "left", "uV")

    # Fill with interpolation
    filled = emg.fillna()

    # Verify
    assert not np.any(np.isnan(filled.to_numpy())), "There should be no NaNs after fillna"
    assert filled.muscle_name == "biceps", "Muscle name should be preserved"
    assert filled.side == "left", "Side should be preserved"
    print("✓ EMGSignal fillna works correctly")


def test_fillna_on_record():
    """Test fillna on Record containing multiple Timeseries"""
    print("\n=== Test fillna on Record ===")

    # Create multiple timeseries with NaNs
    index = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    sig1_data = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    sig1 = Signal1D(sig1_data, index, "V")

    sig2_data = np.array([[1.0, 2.0, 3.0], [np.nan, 2.1, 3.1], [1.2, 2.2, 3.2], [1.3, np.nan, 3.3], [1.4, 2.4, 3.4]])
    sig2 = Signal3D(sig2_data, index, "m")

    # Create Record
    record = Record(signal1=sig1, signal2=sig2)

    # Fill with interpolation
    filled_record = record.fillna()

    # Verify
    assert not np.any(np.isnan(filled_record["signal1"].to_numpy())), "Signal1 should have no NaNs"
    assert not np.any(np.isnan(filled_record["signal2"].to_numpy())), "Signal2 should have no NaNs"
    print("✓ Record fillna works correctly")


def test_fillna_inplace():
    """Test fillna with inplace=True"""
    print("\n=== Test fillna inplace ===")

    # Create test data
    data = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
    index = np.array([0.0, 1.0, 2.0])
    columns = ["col1", "col2"]

    ts = Timeseries(data, index, columns, "m")

    # Verify NaN exists
    assert np.any(np.isnan(ts.to_numpy())), "Original data should contain NaNs"

    # Fill inplace
    result = ts.fillna(value=0.0, inplace=True)

    # Verify
    assert result is None, "Inplace fillna should return None"
    assert not np.any(np.isnan(ts.to_numpy())), "Original object should be modified"
    # Verify the specific value was replaced
    assert ts.to_numpy()[1, 0] == 0.0, "NaN should be replaced with 0.0"
    print("✓ Inplace fillna works correctly")


def test_fillna_with_real_data():
    """Test fillna with real TDF data"""
    print("\n=== Test fillna with real data ===")

    # definizione delle directory di lavoro
    DATA_PATH = join(
        abspath(dirname(__file__)),
        "assets",
        "fillna_data",
        "test_marker.tdf",
    )

    # get the test
    test = laban.WholeBody.from_tdf(filename=DATA_PATH)

    # considero solo i primi 10 secondi
    test = test.loc(0, 10)

    # valuto l'uso dei regressori per il fillna
    regressors = np.concatenate([test[i].to_numpy() for i in ['right_psis', 'left_psis', 'right_asis']], axis=1)
    original = test['left_asis'].copy()
    test['left_asis'] = test['left_asis'].fillna(regressors=regressors)
    filled = test['left_asis']

    # Verify
    assert not np.any(np.isnan(filled.to_numpy())), "Filled data should have no NaNs"
    print("✓ Real data fillna works correctly")

    # Create visualization
    fig = go.Figure()
    for col in ["X", "Y", "Z"]:
        fig.add_trace(
            go.Scatter(
                x=original.index,
                y=original[col].to_numpy().flatten(),
                name=f"{col} - original",
                mode='lines',
                line=dict(width=2)
            )
        )

        # Fillna (tratteggiata)
        fig.add_trace(
            go.Scatter(
                x=filled.index,
                y=filled[col].to_numpy().flatten(),
                name=f"{col} - riempito",
                mode='lines',
                line=dict(width=2, dash='dash')
            )
        )

    fig.update_layout(
        title="Confronto segnale originale vs fillna",
        xaxis_title="Frame",
        yaxis_title="Valore",
        template="simple_white"
    )
    fig_path = DATA_PATH.replace(".tdf", ".html")
    fig.write_html(fig_path)
    print(f"  Figura salvata in {fig_path}")


def run_all_tests():
    """Run all fillna tests"""
    print("\n" + "="*60)
    print("RUNNING FILLNA TESTS AT ALL LEVELS")
    print("="*60)

    try:
        test_fillna_with_constant_value()
        test_fillna_with_interpolation()
        test_fillna_with_regressors()
        test_fillna_on_point3d()
        test_fillna_on_emgsignal()
        test_fillna_on_record()
        test_fillna_inplace()
        test_fillna_with_real_data()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
