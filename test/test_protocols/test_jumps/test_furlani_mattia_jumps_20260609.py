"""analisi dati raccolti"""

# %% SETUP
import subprocess
import sys
from os import makedirs
from os.path import abspath, dirname, join, exists
import warnings

sys.path.append(dirname(dirname(dirname(dirname(abspath(__file__))))))

from src import labanalysis as laban

PARTICIPANT = laban.Participant(
    "Furlani",
    "Mattia",
    "Male",
    184,
    67.8,
    21,
)

RAW_DATA_PATH = join(
    dirname(dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))),
    "t-lab",
    "test",
    "furlani_mattia",
    "2026_06_29",
    "collected_data",
    "jumps",
    "tracked_data",
)


def test_mattia_furlani_jump_tests():

    RESULTS_PATH = "_results"
    TEST_FILE = join(RESULTS_PATH, "test.jumptest")
    if not exists(TEST_FILE):
        makedirs(RESULTS_PATH, exist_ok=True)

        # cmj files path
        cmj_files = [f"counter_movement_jump_{i + 1}" for i in range(3)]
        cmj_files = [join(RAW_DATA_PATH, i + ".tdf") for i in cmj_files]

        # cmj free_hands files path
        cmj_fh_files = [f"counter_movement_jump_free_hands_{i + 1}" for i in range(3)]
        cmj_fh_files = [join(RAW_DATA_PATH, i + ".tdf") for i in cmj_fh_files]

        # list of cmj files with mask for standard and free-hands jumps
        cmj_files += cmj_fh_files
        cmj_fh_mask = [False] * 3 + [True] * 3

        # squat jumps files path
        sj_files = [f"squat_jump_{i + 1}" for i in range(3)]
        sj_files = [join(RAW_DATA_PATH, i + ".tdf") for i in sj_files]

        # generate the test
        test = laban.JumpTest.from_files(
            participant=PARTICIPANT,
            # squat_jump_files=sj_files,
            counter_movement_jump_files=cmj_files,
            counter_movement_jump_free_hands=cmj_fh_mask,
            s2="S2",
            left_foot_ground_reaction_force="left_frz",
            right_foot_ground_reaction_force="right_frz",
            right_psis="RPSI",
            left_psis="LPSI",
            right_asis="RASI",
            left_asis="LASI",
            right_trochanter="RTROC",
            left_trochanter="LTROC",
            right_knee_lateral="RLKN",
            left_knee_lateral="LLKN",
            right_knee_medial="RMKN",
            left_knee_medial="LMKN",
            right_ankle_lateral="RLAN",
            left_ankle_lateral="LLAN",
            right_ankle_medial="RMAN",
            left_ankle_medial="LMAN",
            right_heel="RHEE",
            left_heel="LHEE",
            right_fifth_metatarsal_head="RLMET",
            left_fifth_metatarsal_head="LLMET",
            right_first_metatarsal_head="RMMET",
            left_first_metatarsal_head="LMMET",
            right_toe="RTOE",
            left_toe="LTOE",
        )

        # save the test
        test.save(TEST_FILE, force_overwrite=True)

    # read the test data
    test = laban.JumpTest.load(TEST_FILE)
    if not isinstance(test, laban.JumpTest):
        raise RuntimeError("'Something went wrong storing the test results.")

    # get the results
    results = test.get_results(include_emg=False)
    results.save_all(RESULTS_PATH, force_overwrite=True)

    # remove temporary files
    try:
        subprocess.run(f"rmdir /S /Q '{RESULTS_PATH}'", shell=True)
    except Exception:
        warnings.warn(
            f"Failed to remove temporary files. Please remove them manually. from {RESULTS_PATH}"
        )


if __name__ == "__main__":
    test_mattia_furlani_jump_tests()
