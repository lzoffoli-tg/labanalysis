"""Agility Test analysis"""

import sys
from datetime import date
from os.path import abspath, dirname, join

import numpy as np

# add project root to path like other tests do
sys.path.append(dirname(dirname(abspath(__file__))))

import src.labanalysis as laban


def test_dallavalle_enrico():
    # definizione delle directory di lavoro
    DATA_PATH = join(
        abspath(dirname(__file__)),
        "assets",
        "shuttle_test_data",
        "dallavalle_enrico",
    )

    SHUTTLE_DATA_PATH = join(DATA_PATH, "raw_data")
    RESULTS_DATA_PATH = join(DATA_PATH, "results")

    # generazione dell'utente
    PARTICIPANT = laban.Participant(
        "Dalla Valle",
        "Enrico",
        "Male",
        196,
        birthdate=date(1998, 3, 21),
        recordingdate=date(2026, 1, 21),
    )
    
    #carico i file
    files = laban.get_files(SHUTTLE_DATA_PATH, ".tdf")

    # get the test
    test = laban.ShuttleTest.from_files(
        participant=PARTICIPANT,
        left_foot_ground_reaction_force="Lfoot_frz",
        right_foot_ground_reaction_force="Rfoot_frz",
        filenames=files,
        s2 = "S2",
    )

    # save and load
    filename = join(RESULTS_DATA_PATH, "test.shuttletest")
    test.save(filename, True)
    test = laban.ShuttleTest.load(filename)
    if not isinstance(test, laban.ShuttleTest):
        raise RuntimeError("ShuttleTest loading faild.")

    # get the results
    test.get_results().save_all(RESULTS_DATA_PATH, True)


if __name__ == "__main__":
    test_dallavalle_enrico()
