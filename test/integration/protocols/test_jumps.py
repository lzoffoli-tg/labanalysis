"""Jump Test analysis for Mattia Furlani"""

import sys
from datetime import date
from os.path import abspath, dirname, join

import numpy as np

# add project root to path like other tests do
sys.path.append(dirname(dirname(abspath(__file__))))

import src.labanalysis as laban


def test_furlani_mattia():
    # definizione delle directory di lavoro
    DATA_PATH = join(
        abspath(dirname(__file__)),
        "assets",
        "jumptest_data",
        "furlani_mattia",
    )
    JUMP_DATA_PATH = join(DATA_PATH, "jump_tests")
    BASELINE_DATA_PATH = join(DATA_PATH, "baseline")
    MVC_DATA_PATH = join(DATA_PATH, "mvc")
    RESULTS_DATA_PATH = join(DATA_PATH, "results")

    # generazione dell'utente
    PARTICIPANT = laban.Participant(
        "Mattia",
        "Furlani",
        "Male",
        181,
        birthdate=date(2005, 2, 7),
        recordingdate=date(2025, 12, 18),
    )

    # aggiungo il peso
    BASELINE_FILE = join(BASELINE_DATA_PATH, "baseline.tdf")
    baseline_data = laban.TimeseriesRecord.from_tdf(BASELINE_FILE)
    WEIGHT = baseline_data.forceplatforms.strip(axis=0)
    if not isinstance(WEIGHT, laban.TimeseriesRecord):
        raise ValueError("something went wrong")
    WEIGHT = WEIGHT.resultant_force.force[baseline_data.vertical_axis].to_numpy()
    WEIGHT = float(np.nanmean(WEIGHT) / laban.G)
    PARTICIPANT.set_weight(WEIGHT)

    # Baseline EMG data
    BASELINE_EMG = baseline_data.emgsignals.copy()

    # Normalizazion EMG data
    NORMALIZATION_EMG = laban.TimeseriesRecord()

    BICEPS_DX_EMG_FILE = join(MVC_DATA_PATH, "mvc_bicipite_dx.tdf")
    for i, v in laban.TimeseriesRecord.from_tdf(BICEPS_DX_EMG_FILE).items():
        if (
            isinstance(v, laban.EMGSignal)
            and "biceps femoris" in v.muscle_name
            and v.side == "right"
        ):
            NORMALIZATION_EMG[i] = v

    BICEPS_SX_EMG_FILE = join(MVC_DATA_PATH, "mvc_bicipite_sx.tdf")
    for i, v in laban.TimeseriesRecord.from_tdf(BICEPS_SX_EMG_FILE).items():
        if (
            isinstance(v, laban.EMGSignal)
            and "biceps femoris" in v.muscle_name
            and v.side == "left"
        ):
            NORMALIZATION_EMG[i] = v

    TIBIALE_DX_EMG_FILE = join(MVC_DATA_PATH, "mvc_tibiale_dx.tdf")
    for i, v in laban.TimeseriesRecord.from_tdf(TIBIALE_DX_EMG_FILE).items():
        if (
            isinstance(v, laban.EMGSignal)
            and "tibialis anterior" in v.muscle_name
            and v.side == "right"
        ):
            NORMALIZATION_EMG[i] = v

    TIBIALE_SX_EMG_FILE = join(MVC_DATA_PATH, "mvc_tibiale_sx.tdf")
    for i, v in laban.TimeseriesRecord.from_tdf(TIBIALE_SX_EMG_FILE).items():
        if (
            isinstance(v, laban.EMGSignal)
            and "tibialis anterior" in v.muscle_name
            and v.side == "left"
        ):
            NORMALIZATION_EMG[i] = v

    VASTO_DX_EMG_FILE = join(MVC_DATA_PATH, "mvc_vasto_dx.tdf")
    for i, v in laban.TimeseriesRecord.from_tdf(VASTO_DX_EMG_FILE).items():
        if (
            isinstance(v, laban.EMGSignal)
            and "vastus lateralis" in v.muscle_name
            and v.side == "right"
        ):
            NORMALIZATION_EMG[i] = v

    VASTO_SX_EMG_FILE = join(MVC_DATA_PATH, "mvc_vasto_dx.tdf")
    for i, v in laban.TimeseriesRecord.from_tdf(VASTO_SX_EMG_FILE).items():
        if (
            isinstance(v, laban.EMGSignal)
            and "vastus lateralis" in v.muscle_name
            and v.side == "right"
        ):
            k = i.replace("right", "left")
            r = v.copy()
            r.set_side("left")
            NORMALIZATION_EMG[k] = r

    SOLEO_DX_EMG_FILE = join(MVC_DATA_PATH, "mvc_soleo.tdf")
    for i, v in laban.TimeseriesRecord.from_tdf(SOLEO_DX_EMG_FILE).items():
        if (
            isinstance(v, laban.EMGSignal)
            and "soleus" in v.muscle_name
            and v.side == "right"
        ):
            NORMALIZATION_EMG[i] = v

    SOLEO_SX_EMG_FILE = join(MVC_DATA_PATH, "mvc_soleo.tdf")
    for i, v in laban.TimeseriesRecord.from_tdf(SOLEO_SX_EMG_FILE).items():
        if (
            isinstance(v, laban.EMGSignal)
            and "soleus" in v.muscle_name
            and v.side == "left"
        ):
            NORMALIZATION_EMG[i] = v

    # salti
    sj_files = []
    name_sj_files = [f"squatjump_{i + 1}.tdf" for i in range(3)]
    for i in name_sj_files:
        sj_files.append(join(JUMP_DATA_PATH, i))
    sj_freehands = [False, False, False]

    cmj_files = []
    name_cmj_files = [f"countermovementjump_{i + 1}.tdf" for i in range(3)]
    for i in name_cmj_files:
        cmj_files.append(join(JUMP_DATA_PATH, i))
    cmj_freehands = [False, False, False]

    name_rj_files = [
        "repeatedjumps_freelegs_dx.tdf",
        "repeatedjumps_freelegs_sx.tdf",
        "repeatedjumps_straightlegs_dx.tdf",
        "repeatedjumps_straightlegs_sx.tdf",
        "repeatedjumps_straightlegs_bilateral.tdf",
    ]
    rj_files = []
    for i in name_rj_files:
        rj_files.append(join(JUMP_DATA_PATH, i))
    rj_freehands = [False, False, False, False, False]
    rj_straightlegs = [False, False, True, True, True]
    exclude_repeated_jumps = [[0] for _ in name_rj_files]

    name_dj_files = [
        "dropjump_20cm_bipodalico_1.tdf",
        "dropjump_20cm_bipodalico_2.tdf",
        "dropjump_20cm_bipodalico_3.tdf",
        "dropjump_20cm_dx_1.tdf",
        "dropjump_20cm_dx_2.tdf",
        "dropjump_20cm_sx_1.tdf",
        "dropjump_20cm_sx_2.tdf",
        "dropjump_40cm_bipodalico_1.tdf",
        "dropjump_40cm_bipodalico_2.tdf",
        "dropjump_40cm_dx_1.tdf",
        "dropjump_40cm_dx_2.tdf",
        "dropjump_40cm_sx_1.tdf",
        "dropjump_40cm_sx_2.tdf",
        "dropjump_60cm_bipodalico_1.tdf",
    ]
    dj_files = []
    for i in name_dj_files:
        dj_files.append(join(JUMP_DATA_PATH, i))
    dj_freehands = [True for i in range(len(dj_files))]
    dj_boxheight = [20, 20, 20, 20, 20, 20, 20, 40, 40, 40, 40, 40, 40, 60]

    # get the test
    test = laban.JumpTest.from_files(
        participant=PARTICIPANT,
        emg_normalization_references=NORMALIZATION_EMG,
        emg_normalization_function=np.mean,
        emg_activation_references=BASELINE_EMG,
        left_foot_ground_reaction_force="left_frz",
        right_foot_ground_reaction_force="right_frz",
        squat_jump_files=sj_files,
        squat_jump_free_hands=sj_freehands,
        counter_movement_jump_files=cmj_files,
        counter_movement_jump_free_hands=cmj_freehands,
        repeated_jumps_files=rj_files,
        exclude_repeated_jumps=exclude_repeated_jumps,
        repeated_jumps_straight_leg=rj_straightlegs,
        repeated_jumps_free_hands=rj_freehands,
        drop_jump_files=dj_files,
        drop_jump_free_hands=dj_freehands,
        drop_jump_heights_cm=dj_boxheight,
    )

    # save and load
    filename = join(RESULTS_DATA_PATH, "test.jumptest")
    test.save(filename)
    test = laban.JumpTest.load(filename)
    if not isinstance(test, laban.JumpTest):
        raise RuntimeError("JumpTest loading faild.")

    # get the results
    test.get_results(include_emg=True).save_all(RESULTS_DATA_PATH, True)


def test_dallavalle_enrico():
    # definizione delle directory di lavoro
    DATA_PATH = join(
        abspath(dirname(__file__)),
        "assets",
        "jumptest_data",
        "dallavalle_enrico",
    )
    JUMP_DATA_PATH = join(DATA_PATH, "jumps")
    BASELINE_DATA_PATH = join(DATA_PATH, "baseline")
    MVC_DATA_PATH = join(DATA_PATH, "mvc")
    RESULTS_DATA_PATH = join(DATA_PATH, "results")

    # ottengo il peso
    BASELINE_FILE = join(BASELINE_DATA_PATH, "baseline.tdf")
    baseline_data = laban.TimeseriesRecord.from_tdf(BASELINE_FILE)
    WEIGHT = baseline_data.forceplatforms.strip(axis=0)
    if not isinstance(WEIGHT, laban.TimeseriesRecord):
        raise ValueError("something went wrong")
    WEIGHT = WEIGHT.resultant_force.force[baseline_data.vertical_axis].to_numpy()
    WEIGHT = float(np.nanmean(WEIGHT) / laban.G)

    # generazione dell'utente
    PARTICIPANT = laban.Participant(
        gender="Male",
        weight=WEIGHT,
        recordingdate=date(2026, 1, 21),
    )

    # Normalizazion EMG data
    NORMALIZATION_EMG = laban.TimeseriesRecord()

    TIBIALE_DX_EMG_FILE = join(MVC_DATA_PATH, "mvc_tibiale_dx.tdf")
    for i, v in laban.TimeseriesRecord.from_tdf(TIBIALE_DX_EMG_FILE).items():
        if (
            isinstance(v, laban.EMGSignal)
            and "tibialis anterior" in v.muscle_name
            and v.side == "right"
        ):
            NORMALIZATION_EMG[i] = v

    TIBIALE_SX_EMG_FILE = join(MVC_DATA_PATH, "mvc_tibiale_sx.tdf")
    for i, v in laban.TimeseriesRecord.from_tdf(TIBIALE_SX_EMG_FILE).items():
        if (
            isinstance(v, laban.EMGSignal)
            and "tibialis anterior" in v.muscle_name
            and v.side == "left"
        ):
            NORMALIZATION_EMG[i] = v

    SOLEO_EMG_FILE = join(MVC_DATA_PATH, "mvc_soleo.tdf")
    for i, v in laban.TimeseriesRecord.from_tdf(SOLEO_EMG_FILE).items():
        if isinstance(v, laban.EMGSignal) and "soleus" in v.muscle_name:
            NORMALIZATION_EMG[i] = v

    # salti
    name_dj_files = [
        "dropjump_50cm_bipodalico_1.tdf",
        "dropjump_50cm_bipodalico_2.tdf",
        "dropjump_50cm_bipodalico_3.tdf",
        "dropjump_50cm_dx_1.tdf",
        "dropjump_50cm_dx_2.tdf",
        "dropjump_50cm_dx_3.tdf",
        "dropjump_50cm_sx_1.tdf",
        "dropjump_50cm_sx_2.tdf",
    ]
    dj_files = []
    for i in name_dj_files:
        dj_files.append(join(JUMP_DATA_PATH, i))
    dj_freehands = [True for i in range(len(dj_files))]
    dj_boxheight = [50 for file in dj_files]

    # get the test
    test = laban.JumpTest.from_files(
        participant=PARTICIPANT,
        emg_normalization_references=NORMALIZATION_EMG,
        emg_normalization_function=np.mean,
        left_foot_ground_reaction_force="left_frz",
        right_foot_ground_reaction_force="right_frz",
        drop_jump_files=dj_files,
        drop_jump_free_hands=dj_freehands,
        drop_jump_heights_cm=dj_boxheight,
        relevant_muscle_map = ['soleus', 'tibialis anterior'],
    )

    # save and load
    filename = join(RESULTS_DATA_PATH, "test.jumptest")
    test.save(filename, True)
    test = laban.JumpTest.load(filename)
    if not isinstance(test, laban.JumpTest):
        raise RuntimeError("JumpTest loading faild.")

    # get the results
    test.get_results(include_emg=True).save_all(RESULTS_DATA_PATH, True)


if __name__ == "__main__":
    test_furlani_mattia()
    test_dallavalle_enrico()
