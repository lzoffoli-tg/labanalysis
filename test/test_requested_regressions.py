import numpy as np
import pytest

from labanalysis import (
    Brzycki1RM,
    Converter,
    FreeWeightExercise,
    FreeWeightRepetition,
    Participant,
    Point3D,
    Timeseries,
)


def make_timeseries(data):
    return Timeseries(data, np.arange(len(data)), ["A", "B"], "m")


def test_converter_rejects_unsupported_template(tmp_path):
    notebook = tmp_path / "input.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid template"):
        Converter(notebook).to_html(template="unsupported")


def test_brzycki_rejects_zero_repetitions():
    with pytest.raises(ValueError, match="\\[1, 36\\]"):
        Brzycki1RM().predict_1rm(0, 100)


def test_participant_string_represents_dataframe():
    participant = Participant(name="Ada", surname="Lovelace")

    assert "Ada" in str(participant)
    assert "Lovelace" in str(participant)


@pytest.mark.parametrize("exercise_type", [FreeWeightExercise, FreeWeightRepetition])
def test_freeweight_constructors_preserve_head_markers(exercise_type):
    marker = Point3D([[1, 2, 3]], [0])

    exercise = exercise_type(total_load_kg=10, head_anterior=marker)

    assert exercise.get("head_anterior") is marker


def test_timeseries_drop_removes_named_column_and_rejects_missing():
    timeseries = make_timeseries([[1, 2], [3, 4]])

    dropped = timeseries.drop("B")

    assert dropped.columns.tolist() == ["A"]
    np.testing.assert_array_equal(dropped.to_numpy(), [[1], [3]])
    with pytest.raises(KeyError):
        timeseries.drop("missing")


def test_timeseries_drop_empty_list_keeps_all_columns():
    timeseries = make_timeseries([[1, 2], [3, 4]])

    assert timeseries.drop([]).columns.tolist() == ["A", "B"]


def test_timeseries_delegated_same_shape_method_uses_result():
    timeseries = make_timeseries([[1.234, 2.345], [3.456, 4.567]])

    rounded = timeseries.round(1)

    np.testing.assert_array_equal(rounded.to_numpy(), [[1.2, 2.3], [3.5, 4.6]])
    np.testing.assert_array_equal(
        timeseries.to_numpy(), [[1.234, 2.345], [3.456, 4.567]]
    )


@pytest.mark.parametrize(
    ("axis", "expected_shape"),
    [(None, (0, 0)), (0, (0, 2)), (1, (2, 0))],
)
def test_timeseries_strip_handles_all_nan_data(axis, expected_shape):
    timeseries = make_timeseries([[np.nan, np.nan], [np.nan, np.nan]])

    assert timeseries.strip(axis=axis).shape == expected_shape
