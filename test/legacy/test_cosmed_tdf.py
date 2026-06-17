"""Tests for SubmaximalVO2MaxTest"""

import sys
from os.path import abspath, dirname, join

sys.path.append(dirname(dirname(abspath(__file__))))

import src.labanalysis as laban

# definizione delle directory di lavoro
PATH = join(
    abspath(dirname(__file__)),
    "assets",
    "cosmed_data",
)


def test_csv_reading():
    # get the test
    file = join(PATH, "test_csv.csv")
    p = laban.Participant.from_cosmed_file(file)
    test = laban.SubmaximalVO2MaxTest.from_files(
        participant=p,
        filename=file,
        breath_by_breath=False,
    )
    assert isinstance(test, laban.SubmaximalVO2MaxTest)


def test_excel_reading():
    # get the test
    file = join(PATH, "test_excel.xlsx")
    p = laban.Participant.from_cosmed_file(file)
    test = laban.SubmaximalVO2MaxTest.from_files(
        participant=p,
        filename=file,
        breath_by_breath=True,
    )
    assert isinstance(test, laban.SubmaximalVO2MaxTest)
