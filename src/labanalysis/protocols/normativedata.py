"""normative data module"""

import pandas as pd

__all__ = [
    "isok_1rm_normative_values",
    "uprightbalance_normative_values",
    "plankbalance_normative_values",
    "jumps_normative_values",
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

jumps_normative_values = pd.DataFrame(
    [
        ["repeated counter movement jumps", "bilateral", "rsi (cm/s)", 1.1, 0.47],
        ["squat jump", "bilateral", "elevation (cm)", 35.0, 9.4],
        ["counter movement jump", "bilateral", "elevation (cm)", 36.5, 10.5],
        ["counter movement jump free hands", "bilateral", "elevation (cm)", 41.5, 12.1],
        ["counter movement jump", "unilateral", "elevation (cm)", 19.0, 6.2],
        ["drop jump (40cm)", "bilateral", "rsi (cm/s)", 1.2, 0.3],
        ["drop jump (40cm)", "bilateral", "elevation (cm)", 28.0, 6.0],
        ["drop jump (40cm)", "bilateral", "contact time (ms)", 235, 34],
        ["drop jump (40cm)", "unilateral", "rsi (cm/s)", 0.43, 0.12],
        ["drop jump (40cm)", "unilateral", "elevation (cm)", 14.4, 3.1],
        ["drop jump (40cm)", "unilateral", "contact time (ms)", 350, 60],
    ],
    columns=["type", "side", "parameter", "mean", "std"],
)
