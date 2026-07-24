"""analisi dati raccolti"""

# %% SETUP
import shutil
import sys
from os import makedirs
from os.path import abspath, dirname, exists, join, sep
from pathlib import Path

sys.path.append(dirname(dirname(dirname(dirname(abspath(__file__))))))

from src import labanalysis as laban


def test_erika_furlani_jump_tests():

    PARTICIPANT = laban.Participant(
        "Furlani",
        "Erika",
        "Female",
        176,
        57.6,
        30,
    )

    RAW_DATA_PATH = Path(
        join(
            dirname(dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))),
            "t-lab",
            "test",
            "furlani_erika",
            "2026_06_29",
            "collected_data",
            "tracked",
            "drop_jumps",
        )
    )

    RESULTS_PATH = Path(__file__).parent / "results"
    TEST_FILE = RESULTS_PATH / "test.jumptest"
    if not exists(TEST_FILE):

        # get the files
        dropjump_files = []
        box_height = []
        free_hands = []
        for file in RAW_DATA_PATH.glob("*.tdf"):
            name = str(file).rsplit(sep, 1)[-1].rsplit(".", 1)[0]
            parts = name.split("_")
            jump_type = "_".join(parts[:2])
            height = parts[2]
            if jump_type != "drop_jump":
                continue
            dropjump_files.append(file)
            box_height.append(int(height))
            free_hands.append(True)

        # get the test
        test_dj = laban.JumpTest.from_files(
            participant=PARTICIPANT,
            left_foot_ground_reaction_force="left_frz",
            right_foot_ground_reaction_force="right_frz",
            s2="S2",
            drop_jump_files=dropjump_files,
            drop_jump_heights_cm=box_height,
            drop_jump_free_hands=free_hands,
        )

        # save the test
        test_dj.save(TEST_FILE, force_overwrite=True)

    # read the test data
    test_dj = laban.JumpTest.load(TEST_FILE)
    if not isinstance(test_dj, laban.JumpTest):
        raise RuntimeError("'Something went wrong storing the test results.")

    # for jump in test_dj.drop_jumps:
    # jump.box_height = jump.box_height_cm

    # get the results
    results_dj = test_dj.get_results(include_emg=False)
    results_dj.save_all(RESULTS_PATH, force_overwrite=True)


if __name__ == "__main__":
    test_erika_furlani_jump_tests()
