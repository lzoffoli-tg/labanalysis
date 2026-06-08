"""Tests for Isokinetic1RMTest and Isokinetic1RMTestResults"""

import sys
from os.path import abspath, dirname, join, exists

sys.path.append(dirname(dirname(abspath(__file__))))

from datetime import date

import numpy as np

import src.labanalysis as laban


# definizione delle directory di lavoro
PATH = join(
    abspath(dirname(__file__)),
    "assets",
    "isokinetic1rmtest_data",
    "leclerc_charles",
)
RESULTS_PATH = join(PATH, "results")
ISOKINETIC_DATA_PATH = join(PATH, "isokinetic1rmtest_data")
BASELINE_DATA_PATH = join(PATH, "baseline")
MVC_DATA_PATH = join(PATH, "mvc")

# generazione dell'utente
PARTICIPANT = laban.Participant(
    "Charles",
    "Leclerc",
    "Male",
    179,
    birthdate=date(1997, 10, 16),
    recordingdate=date(2026, 1, 14),
)

# aggiungo il peso
BASELINE_FILE = join(BASELINE_DATA_PATH, "baseline.tdf")
baseline_data = laban.TimeseriesRecord.from_tdf(BASELINE_FILE)
WEIGHT = baseline_data.forceplatforms.strip(axis=0)
if not isinstance(WEIGHT, laban.TimeseriesRecord):
    raise ValueError("something went wrong")
WEIGHT = WEIGHT.resultant_force.force[baseline_data.vertical_axis]
WEIGHT = float(np.nanmean(WEIGHT.to_numpy()) / laban.G)
PARTICIPANT.set_weight(WEIGHT)

# Baseline EMG data
BASELINE_EMG = baseline_data.emgsignals.copy()

# Normalizazion EMG data
NORMALIZATION_EMG = laban.TimeseriesRecord()

VASTO_DX_EMG_FILE = join(MVC_DATA_PATH, "mvc_vasto_dx.tdf")
for i, v in laban.TimeseriesRecord.from_tdf(VASTO_DX_EMG_FILE).items():
    if (
        isinstance(v, laban.EMGSignal)
        and "vastus lateralis" in v.muscle_name
        and v.side == "right"
    ):
        NORMALIZATION_EMG[i] = v


VASTO_SX_EMG_FILE = join(MVC_DATA_PATH, "mvc_vasto_sx.tdf")
for i, v in laban.TimeseriesRecord.from_tdf(VASTO_SX_EMG_FILE).items():
    if (
        isinstance(v, laban.EMGSignal)
        and "vastus lateralis" in v.muscle_name
        and v.side == "left"
    ):
        NORMALIZATION_EMG[i] = v

BICEPS_DX_EMG_FILE = join(MVC_DATA_PATH, "mvc_bicipite_dx.tdf")
for i, v in laban.TimeseriesRecord.from_tdf(BICEPS_DX_EMG_FILE).items():
    if (
        isinstance(v, laban.EMGSignal)
        and "biceps femoris" in v.muscle_name
        and v.side == "right"
    ):
        NORMALIZATION_EMG[i] = v

BICEPS_SX_EMG_FILE = join(MVC_DATA_PATH, "mvc_bicipite_sx.tdf")
for i, v in laban.TimeseriesRecord.from_tdf(BICEPS_SX_EMG_FILE).items():
    if (
        isinstance(v, laban.EMGSignal)
        and "biceps femoris" in v.muscle_name
        and v.side == "left"
    ):
        NORMALIZATION_EMG[i] = v

PETTORALE_DX_EMG_FILE = join(MVC_DATA_PATH, "mvc_pettorale_dx.tdf")
for i, v in laban.TimeseriesRecord.from_tdf(PETTORALE_DX_EMG_FILE).items():
    if (
        isinstance(v, laban.EMGSignal)
        and "pectoralis major" in v.muscle_name
        and v.side == "right"
    ):
        NORMALIZATION_EMG[i] = v

PETTORALE_SX_EMG_FILE = join(MVC_DATA_PATH, "mvc_pettorale_sx.tdf")
for i, v in laban.TimeseriesRecord.from_tdf(PETTORALE_SX_EMG_FILE).items():
    if (
        isinstance(v, laban.EMGSignal)
        and "pectoralis major" in v.muscle_name
        and v.side == "left"
    ):
        NORMALIZATION_EMG[i] = v

DORSALE_DX_EMG_FILE = join(MVC_DATA_PATH, "mvc_dorsale_dx.tdf")
for i, v in laban.TimeseriesRecord.from_tdf(DORSALE_DX_EMG_FILE).items():
    if (
        isinstance(v, laban.EMGSignal)
        and "latissimus dorsi" in v.muscle_name
        and v.side == "right"
    ):
        NORMALIZATION_EMG[i] = v


DORSALE_SX_EMG_FILE = join(MVC_DATA_PATH, "mvc_dorsale_dx.tdf")
for i, v in laban.TimeseriesRecord.from_tdf(VASTO_SX_EMG_FILE).items():
    if (
        isinstance(v, laban.EMGSignal)
        and "latissimus dorsi" in v.muscle_name
        and v.side == "left"
    ):
        NORMALIZATION_EMG[i] = v


def test_leclerc_isokinetic1rm():
    test_muscle_map = {
        "CHEST PRESS": ["pectoralis major"],
        "LEG PRESS REV": ["vastus lateralis", "biceps femoris"],
        "LOW ROW": ["latissimus dorsi"],
    }
    for product, muscle_map in test_muscle_map.items():

        # get the test
        product_str = product.replace(" ", "").lower()
        product_name = product.replace(" ", "").lower()
        destination_path = join(RESULTS_PATH, product_name)
        isokinetic_path = join(PATH, "isokinetic_tests")

        left_root = join(isokinetic_path, "_".join([product_str, "sx"]))
        right_root = join(isokinetic_path, "_".join([product_str, "dx"]))
        bilateral_root = join(isokinetic_path, "_".join([product_str, "bilateral"]))
        left_txt = left_root + ".txt"
        left_tdf = left_root + ".tdf"
        right_txt = right_root + ".txt"
        right_tdf = right_root + ".tdf"
        bilateral_txt = bilateral_root + ".txt"
        bilateral_tdf = bilateral_root + ".tdf"

        test = laban.Isokinetic1RMTest.from_files(
            participant=PARTICIPANT,
            product=product,  # type: ignore
            emg_normalization_references=NORMALIZATION_EMG,
            emg_normalization_function=np.max,
            emg_activation_references=BASELINE_EMG,
            emg_activation_threshold=5,
            left_biostrength_filename=left_txt if exists(left_txt) else None,
            left_emg_filename=left_tdf if exists(left_tdf) else None,
            right_biostrength_filename=right_txt if exists(right_txt) else None,
            right_emg_filename=right_tdf if exists(right_tdf) else None,
            bilateral_biostrength_filename=(
                bilateral_txt if exists(bilateral_txt) else None
            ),
            bilateral_emg_filename=bilateral_tdf if exists(bilateral_tdf) else None,
            relevant_muscle_map=muscle_map,
        )

        # save the test
        destination_file = join(destination_path, "test.isokinetic1rmtest")
        test.save(destination_file, force_overwrite=True)

        # read the test data
        test = laban.Isokinetic1RMTest.load(destination_file)
        if not isinstance(test, laban.TestProtocol):
            raise RuntimeError("'Something went wrong storing the test results.")

        # get the results
        res = test.get_results()

        # save all
        res.save_all(destination_path)
