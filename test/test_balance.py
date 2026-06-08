"""jump analysis testing for SingleJump, JumpExercise, and DropJump"""

import sys
import itertools
from datetime import datetime
from os.path import abspath, dirname, join
from typing import Literal

import numpy as np

# add project root to path like other tests do
sys.path.append(dirname(dirname(abspath(__file__))))

from src import labanalysis as laban


def _check_balance_test(
    mode: Literal["stabilità", "plank"] | str,
    side: Literal["bilaterale", "sx", "dx", ""] | str,
    eyes: Literal["occhiaperti", "occhichiusi"] | str,
    path: str,
):

    # setup
    root = mode if mode == "plank" else "_".join([mode, side])
    filename = "_".join([root, eyes])
    if mode == "stabilità":
        test_fun = laban.UprightBalanceTest
    else:
        test_fun = laban.PlankBalanceTest
    participant = laban.Participant(recordingdate=datetime.now())
    source_file = join(path, filename + ".tdf")
    eyes_lbl = "open" if eyes == "occhiaperti" else "closed"
    forces = dict()
    if mode == "stabilità":
        if side in ["bilaterale", "sx"]:
            forces["left_foot_ground_reaction_force"] = "Lfoot_frz"
        if side in ["bilaterale", "dx"]:
            forces["right_foot_ground_reaction_force"] = "Rfoot_frz"
    else:
        forces = dict(
            left_foot_ground_reaction_force="Lfoot_frz",
            right_foot_ground_reaction_force="Rfoot_frz",
            left_hand_ground_reaction_force="Lhand_frz",
            right_hand_ground_reaction_force="Rhand_frz",
        )

    # get the test
    test = test_fun.from_files(
        participant=participant,
        filename=source_file,
        eyes=eyes_lbl,
        **forces,  # type: ignore
    )

    # save the test
    extension = test_fun.__name__.lower()
    results_path = join(path, "results", filename)
    destination_file = join(results_path, f"test.{extension}")
    test.save(destination_file, False)

    # read the test data
    test = test_fun.load(destination_file)
    if not isinstance(test, laban.TestProtocol):
        raise RuntimeError("'Something went wrong storing the test results.")

    # get the results
    results = test.get_results(include_emg=False)

    # save
    results.save_all(results_path, True)


def test_balance_leclerc():
    path = join(dirname(__file__), "assets", "balance_data", "leclerc_charles")
    pairs = [["stabilità"], ["bilaterale", "dx", "sx"], ["occhiaperti", "occhichiusi"]]
    pairs = list(itertools.product(*pairs))
    pairs += list(itertools.product(["plank"], [""], ["occhiaperti", "occhichiusi"]))
    for mode, side, eyes in pairs:
        print(f"{mode}-{side}-{eyes}".upper())
        _check_balance_test(
            mode,
            side,
            eyes,
            path,
        )
