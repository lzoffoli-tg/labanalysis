"""Tests for SubmaximalVO2MaxTest"""

import sys
from os.path import abspath, dirname, exists, join

sys.path.append(dirname(dirname(abspath(__file__))))

from datetime import date

import numpy as np

import src.labanalysis as laban

"""
# definizione delle directory di lavoro
PATH = join(
    abspath(dirname(__file__)),
    "assets",
    "submaximal_vo2max",
    "leclerc_charles",
)
RESULTS_PATH = join(PATH, "results")

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
BASELINE_FILE = join(PATH, "baseline.tdf")
baseline_data = laban.TimeseriesRecord.from_tdf(BASELINE_FILE)
WEIGHT = baseline_data.forceplatforms.strip(axis=0)
if not isinstance(WEIGHT, laban.TimeseriesRecord):
    raise ValueError("something went wrong")
WEIGHT = WEIGHT.resultant_force.force[baseline_data.vertical_axis]
WEIGHT = float(np.nanmean(WEIGHT.to_numpy()) / laban.G)
PARTICIPANT.set_weight(WEIGHT)


def test_leclerc_vo2max():
    # get the test
    VO2_FILE = join(PATH, "raw_data.csv")
    test = laban.SubmaximalVO2MaxTest.from_files(
        participant=PARTICIPANT,
        filename=VO2_FILE,
    )

    # save the test
    destination_file = join(RESULTS_PATH, "test.submaximalvo2maxtest")
    test.save(destination_file, force_overwrite=True)

    # read the test data
    test = laban.SubmaximalVO2MaxTest.load(destination_file)
    if not isinstance(test, laban.TestProtocol):
        raise RuntimeError("'Something went wrong storing the test results.")

    # get the results
    res = test.get_results()

    # save all
    res.save_all(RESULTS_PATH)
"""

# definizione delle directory di lavoro
# Enrico Dalla Valle
PATH_DVE = join(
    abspath(dirname(__file__)),
    "assets",
    "submaximal_vo2max",
    "dallavalle_enrico",
)
RESULTS_PATH_DVE = join(PATH_DVE, "results")

# generazione dell'utente
PARTICIPANT_DVE = laban.Participant(
    "Dalla Valle",
    "Enrico",
    "Male",
    196,
    birthdate=date(1998, 3, 21),
    recordingdate=date(2026, 1, 21),
)

# aggiungo il peso
BASELINE_FILE_DVE = join(PATH_DVE, "stabilità_bilaterale_occhiaperti.tdf")
baseline_data = laban.TimeseriesRecord.from_tdf(BASELINE_FILE_DVE)
WEIGHT = baseline_data.forceplatforms.strip(axis=0)
if not isinstance(WEIGHT, laban.TimeseriesRecord):
    raise ValueError("something went wrong")
WEIGHT = WEIGHT.resultant_force.force[baseline_data.vertical_axis]
WEIGHT = float(np.nanmean(WEIGHT.to_numpy()) / laban.G)
PARTICIPANT_DVE.set_weight(WEIGHT)


def test_dallavalle_vo2max():
    # get the test
    VO2_FILE_DVE = join(PATH_DVE, "raw_data_dallavalle.csv")
    test = laban.SubmaximalVO2MaxTest.from_files(
        participant=PARTICIPANT_DVE,
        filename=VO2_FILE_DVE,
    )

    # get the participant
    p = laban.Participant.from_cosmed_file(VO2_FILE_DVE)

    # save the test
    destination_file = join(RESULTS_PATH_DVE, "test.submaximalvo2maxtest")
    test.save(destination_file, force_overwrite=True)

    # read the test data
    test = laban.SubmaximalVO2MaxTest.load(destination_file)
    if not isinstance(test, laban.TestProtocol):
        raise RuntimeError("'Something went wrong storing the test results.")

    # get the results
    res = test.get_results()

    # save all
    res.save_all(RESULTS_PATH_DVE, force_overwrite=True)
