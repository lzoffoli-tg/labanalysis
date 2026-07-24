"""test analysis module"""

#! imports
import shutil
import sys
from os.path import abspath, dirname, exists
from pathlib import Path

import numpy as np

sys.path.append(dirname(dirname(dirname(dirname(dirname(abspath(__file__)))))))

from src.labanalysis.constants import G
from src.labanalysis.protocols.gaittests.running_test import RunningTest
from src.labanalysis.protocols.participant import Participant
from src.labanalysis.records.body.wholebody import WholeBody


def test_running_test():
    #! path setup
    FILE_PATH = Path(__file__).absolute()
    TRACKED_PATH = (
        FILE_PATH.parent.parent.parent.parent.parent.parent.parent
        / "t-lab"
        / "test"
        / "crippa_yeman"
        / "2026_06_03"
        / "collected_data"
        / "tracked"
    )
    RESULTS_PATH = FILE_PATH.parent / "results"
    RESULTS_PATH.mkdir(exist_ok=True)
    TEST_FILE = RESULTS_PATH / "running_test.runningtest"

    if not exists(TEST_FILE):

        #! user data
        MARKERSET = dict(
            left_toe="lToe",
            left_fifth_metatarsal_head="lMeta5",
            left_first_metatarsal_head="lMeta1",
            left_heel="lHeel",
            left_ankle_medial="lMalMed",
            left_ankle_lateral="lMalExt",
            left_knee_medial="lKneeMed",
            left_knee_lateral="lKneeExt",
            left_trochanter="lTroc",
            right_toe="rToe",
            right_fifth_metatarsal_head="rMeta5",
            right_first_metatarsal_head="rMeta1",
            right_heel="rHeel",
            right_ankle_medial="rMalMed",
            right_ankle_lateral="rMalExt",
            right_knee_medial="rKneeMed",
            right_knee_lateral="rKneeExt",
            right_trochanter="rTroc",
            left_asis="lASIS",
            right_asis="rASIS",
            left_psis="lPSIS",
            right_psis="rPSIS",
            s2="L2",
            c7="C7",
            right_acromion="rAcro",
            left_acromion="lAcro",
            left_elbow_medial="lElbMed",
            right_elbow_medial="rElbMed",
            left_elbow_lateral="lElbExt",
            right_elbow_lateral="rElbExt",
            left_wrist_medial="lWriMed",
            right_wrist_medial="rWriMed",
            left_wrist_lateral="lWriExt",
            right_wrist_lateral="rWriExt",
            sc="cla",
        )
        BASELINE_FILE = TRACKED_PATH / "baseline.tdf"
        BASELINE = WholeBody.from_tdf(BASELINE_FILE, **MARKERSET)
        BODYWEIGHT = BASELINE.resultant_force.force
        BODYWEIGHT = float(np.nanmean(BODYWEIGHT[BODYWEIGHT.vertical_axis])) / G
        PARTICIPANT = Participant(name="Yeman", surname="Crippa", weight=BODYWEIGHT)

        #! test creation
        TEST_FILES = []
        TEST_SPEEDS = []
        TEST_GRADES = []
        for i in TRACKED_PATH.glob("*.tdf"):
            if "baseline" not in i.name:
                speed, grade = i.name.split("_")
                TEST_SPEEDS.append(float(speed[1:]))
                TEST_GRADES.append(float(grade.split(".")[0][1:]))
                TEST_FILES.append(i.absolute())
        RUNNING_TEST = RunningTest.from_files(
            files=TEST_FILES,
            speeds=TEST_SPEEDS,
            grades=TEST_GRADES,
            algorithm="kinetics",
            participant=PARTICIPANT,
            **MARKERSET,  # type: ignore
        )
        RUNNING_TEST.save(TEST_FILE)

    #! test loading / processing / results extraction / results saving
    RunningTest.load(TEST_FILE).get_results().save_all(RESULTS_PATH)

    # remove temporary files
    shutil.rmtree(RESULTS_PATH, ignore_errors=True)


if __name__ == "__main__":
    test_running_test()
