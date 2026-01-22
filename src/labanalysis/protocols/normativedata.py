"""normative data module"""

import pandas as pd

__all__ = [
    "isok_1rm_normative_values",
    "uprightbalance_normative_values",
    "plankbalance_normative_values",
    "jumps_normative_values",
    "vo2max_normative_values",
]

# ISOK 1RM normative values
isok_1rm_normative_values = pd.DataFrame(
    {
        "parameter": [
            "abductor",
            "adductor",
            "arm curl",
            "arm extension",
            "chest press",
            "leg curl",
            "leg extension",
            "leg press",
            "low row",
            "lower back",
            "pectoral",
            "reverse fly",
            "shoulder press",
            "total abdominal",
            "vertical traction",
        ],
        "mean": [
            12.6,
            7.5,
            7.5,
            4.8,
            18.9,
            11.8,
            19.5,
            51.1,
            15.8,
            14.1,
            8.6,
            10.4,
            10.4,
            10.6,
            18.5,
        ],
        "std": [
            4.5,
            5.6,
            4.3,
            2.3,
            16.3,
            4.1,
            10.3,
            37.2,
            7.8,
            7.3,
            7.3,
            6.4,
            6.0,
            4.5,
            13.6,
        ],
    }
)

# Upright Balance normative values
uprightbalance_normative_values = pd.DataFrame(
    {
        "parameter": ["area_of_stability_mm2"] * 6,
        "side": ["bilateral", "bilateral", "right", "right", "left", "left"],
        "eyes": ["open", "closed", "open", "closed", "open", "closed"],
        "mean": [125, 175, 400, 800, 400, 800],
        "std": [82, 125, 264, 528, 264, 528],
    }
)

# Plank Balance normative values
plankbalance_normative_values = pd.DataFrame(
    {
        "parameter": ["area_of_stability_mm2", "area_of_stability_mm2"],
        "eyes": ["open", "closed"],
        "mean": [125, 175],
        "std": [82, 125],
    }
)

# jumps normative values
_male_jumps_normative_values = pd.DataFrame(
    [
        ["repeated jumps", "Male", "bilateral", "rsi (cm/s)", 110, 47],
        ["repeated jumps", "Male", "bilateral", "elevation (cm)", 36.5, 10.5],
        ["repeated jumps", "Male", "bilateral", "contact time (ms)", 235, 34],
        ["repeated jumps", "Male", "unilateral", "elevation (cm)", 19.0, 6.2],
        ["repeated jumps", "Male", "unilateral", "rsi (cm/s)", 43, 12],
        ["repeated jumps", "Male", "unilateral", "contact time (ms)", 350, 60],
        ["squat jump", "Male", "bilateral", "elevation (cm)", 35.0, 9.4],
        ["free hand jump", "Male", "bilateral", "elevation (cm)", 41.5, 12.1],
        ["counter movement jump", "Male", "bilateral", "elevation (cm)", 36.5, 10.5],
        ["counter movement jump", "Male", "unilateral", "elevation (cm)", 19.0, 6.2],
        ["drop jump (40cm)", "Male", "bilateral", "rsi (cm/s)", 120, 30],
        ["drop jump (40cm)", "Male", "bilateral", "elevation (cm)", 28.0, 6.0],
        ["drop jump (40cm)", "Male", "bilateral", "contact time (ms)", 235, 34],
        ["drop jump (40cm)", "Male", "unilateral", "rsi (cm/s)", 43, 12],
        ["drop jump (40cm)", "Male", "unilateral", "elevation (cm)", 14.4, 3.1],
        ["drop jump (40cm)", "Male", "unilateral", "contact time (ms)", 350, 60],
        ["drop jump (40cm)", "Male", "bilateral", "activation ratio", 100, 25],
        ["drop jump (40cm)", "Male", "unilateral", "activation ratio", 100, 25],
        ["drop jump (40cm)", "Male", "bilateral", "activation time (ms)", -80, 60],
        ["drop jump (40cm)", "Male", "unilateral", "activation time (ms)", -80, 60],
    ],
    columns=["type", "gender", "side", "parameter", "mean", "std"],
)

# create female normative values with 33% less performances (remember to add references)
_female_jumps_normative_values = _male_jumps_normative_values.copy()
_female_jumps_normative_values.gender = _female_jumps_normative_values.gender.map(
    lambda x: "Female"
)
_idx = _female_jumps_normative_values.parameter.map(
    lambda x: x in ["elevation (cm)", "rsi (cm/s)"]
)
_female_jumps_normative_values.loc[_idx, ["mean", "std"]] = (
    _female_jumps_normative_values.loc[_idx, ["mean", "std"]] * 0.67
)
_female_jumps_normative_values = pd.DataFrame(_female_jumps_normative_values)
jumps_normative_values = pd.concat(
    [_male_jumps_normative_values, _female_jumps_normative_values],
    ignore_index=True,
)

# VO2max data (from ACSM)
vo2max_normative_values = pd.DataFrame(
    data=[
        # RUN - Male
        ["RUN", "Male", 20, 29, 38.10, 44.90, 49.00, 55.20],
        ["RUN", "Male", 30, 39, 34.10, 39.60, 43.80, 49.20],
        ["RUN", "Male", 40, 49, 30.50, 35.70, 38.90, 45.00],
        ["RUN", "Male", 50, 59, 26.10, 30.70, 33.80, 39.70],
        ["RUN", "Male", 60, 69, 22.40, 26.60, 29.10, 34.50],
        # RUN - Female
        ["RUN", "Female", 20, 29, 28.60, 34.60, 38.90, 44.70],
        ["RUN", "Female", 30, 39, 24.10, 28.20, 31.20, 36.10],
        ["RUN", "Female", 40, 49, 21.30, 24.90, 27.70, 32.40],
        ["RUN", "Female", 50, 59, 19.10, 21.80, 24.40, 27.60],
        ["RUN", "Female", 60, 69, 16.50, 18.90, 20.50, 23.80],
        # BIKE - Male
        ["BIKE", "Male", 20, 29, 33.20, 38.30, 43.10, 49.50],
        ["BIKE", "Male", 30, 39, 25.40, 28.10, 30.70, 35.00],
        ["BIKE", "Male", 40, 49, 22.20, 25.40, 28.00, 31.80],
        ["BIKE", "Male", 50, 59, 21.50, 23.60, 25.70, 29.30],
        ["BIKE", "Male", 60, 69, 19.00, 21.40, 22.90, 25.50],
        # BIKE - Female
        ["BIKE", "Female", 20, 29, 21.60, 28.10, 32.40, 37.10],
        ["BIKE", "Female", 30, 39, 17.00, 20.10, 22.10, 25.10],
        ["BIKE", "Female", 40, 49, 15.80, 18.40, 20.00, 22.60],
        ["BIKE", "Female", 50, 59, 14.90, 16.60, 17.70, 20.10],
        ["BIKE", "Female", 60, 69, 14.00, 15.40, 16.30, 18.30],
    ],
    columns=["movement", "gender", "age_from", "age_to", "p20", "p40", "p60", "p80"],
)
