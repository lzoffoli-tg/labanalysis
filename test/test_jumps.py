"""jump analysis testing for SingleJump, JumpExercise, and DropJump"""

import sys
from datetime import datetime
from os.path import abspath, dirname, join

import pandas as pd

# add project root to path like other tests do
sys.path.append(dirname(dirname(abspath(__file__))))

from src.labanalysis.records.records import TimeseriesRecord
from src.labanalysis.protocols.jumptests import JumpTest
from src.labanalysis.protocols.normativedata import jumps_normative_values

# import classes under test
from src.labanalysis.protocols.protocols import Participant

participant = Participant(
    weight=84,
    recordingdate=datetime.now(),
    gender="female",
)

cmj_path = join(
    dirname(__file__), "assets", "jumptest_data", "countermovementjump_{}.tdf"
)
sj_path = join(dirname(__file__), "assets", "jumptest_data", "squatjump_{}.tdf")

dj_dx_path = join(dirname(__file__), "assets", "jumptest_data", "drop_jump_dx_{}.tdf")
dj_sx_path = join(dirname(__file__), "assets", "jumptest_data", "drop_jump_sx_{}.tdf")
dj_both_path = join(
    dirname(__file__), "assets", "jumptest_data", "drop_jump_bilaterale_{}.tdf"
)

balance_path = join(dirname(__file__), "assets", "balance_data")
emg_normalization_path = join(balance_path, "stabilità_bilaterale_occhiaperti.tdf")
emg_activation_data = TimeseriesRecord.from_tdf(emg_normalization_path)
emg_activation_data = emg_activation_data.emgsignals


def test_dropjumps():
    dj_files = [dj_dx_path.format(str(i + 1)) for i in range(3)]
    dj_files += [dj_sx_path.format(str(i + 1)) for i in range(3)]
    dj_files += [dj_both_path.format(str(i + 1)) for i in range(3)]
    test = JumpTest.from_files(
        participant=participant,
        normative_data=jumps_normative_values,
        left_foot_ground_reaction_force="LFOOT_FP",
        right_foot_ground_reaction_force="RFOOT_FP",
        drop_jump_height_cm=40,
        emg_activation_references=emg_activation_data,
        emg_activation_threshold=3,
        relevant_muscle_map=None,
        drop_jump_files=dj_files,
    )
    results = test.get_results()
    check = 1
