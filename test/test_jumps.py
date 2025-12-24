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

participant = Participant(weight=60, recordingdate=datetime.now())

cmj_path = join(
    dirname(__file__), "assets", "jumptest_data", "countermovementjump_{}.tdf"
)
sj_path = join(dirname(__file__), "assets", "jumptest_data", "squatjump_{}.tdf")

dj_dx_path = join(dirname(__file__), "assets", "jumptest_data", "dropjump_dx_{}.tdf")
dj_sx_path = join(dirname(__file__), "assets", "jumptest_data", "dropjump_sx_{}.tdf")

emg_baseline_path = join(
    dirname(__file__),
    "assets",
    "balance_data",
    "normalization_data.tdf",
)
emg_norm_data = TimeseriesRecord.from_tdf(emg_baseline_path)
emg_norm_data = emg_norm_data.emgsignals


def test_jumptest():
    test = JumpTest.from_files(
        participant=participant,
        normative_data=jumps_normative_values,
        left_foot_ground_reaction_force="left_frz",
        right_foot_ground_reaction_force="right_frz",
        drop_jump_height_cm=40,
        emg_normalization_references=emg_norm_data,
        emg_activation_references=emg_norm_data,
        emg_activation_threshold=3,
        relevant_muscle_map=None,
        squat_jump_files=[sj_path.format(str(i + 1)) for i in range(2)],
        counter_movement_jump_files=[cmj_path.format(str(i + 1)) for i in range(2)],
        drop_jump_files=[dj_dx_path.format(str(i + 1)) for i in range(1)]
        + [dj_sx_path.format(str(i + 1)) for i in range(1)],
    )
    results = test.results()
    check = 1
