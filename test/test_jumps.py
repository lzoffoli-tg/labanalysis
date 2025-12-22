"""jump analysis testing for SingleJump, JumpExercise, and DropJump"""

import sys
from datetime import datetime
from os.path import abspath, dirname, join

import pandas as pd

# add project root to path like other tests do
sys.path.append(dirname(dirname(abspath(__file__))))

from src.labanalysis.protocols.jumptests import JumpTest

# import classes under test
from src.labanalysis.protocols.protocols import Participant

participant = Participant(weight=60, recordingdate=datetime.now())

cmj_path = join(
    dirname(__file__), "assets", "jumptest_data", "countermovementjump_{}.tdf"
)
sj_path = join(dirname(__file__), "assets", "jumptest_data", "squatjump_{}.tdf")

dj_dx_path = join(dirname(__file__), "assets", "jumptest_data", "dropjump_dx_{}.tdf")
dj_sx_path = join(dirname(__file__), "assets", "jumptest_data", "dropjump_sx_{}.tdf")


def test_jumptest():
    test = JumpTest.from_files(
        participant,
        pd.DataFrame(),
        "left_frz",
        "right_frz",
        30,
        3,
        [sj_path.format(str(i + 1)) for i in range(3)],
        [cmj_path.format(str(i + 1)) for i in range(3)],
        [dj_dx_path.format(str(i + 1)) for i in range(3)]
        + [dj_sx_path.format(str(i + 1)) for i in range(3)],
    )
    results = test.results
    check = 1
