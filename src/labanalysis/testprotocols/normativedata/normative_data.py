import pandas as pd

# CMJ normative values
cmj_normative_values = pd.DataFrame(
    {
        "parameter": [
            "Biceps femoris Imbalance (%)",
            "Elevation (cm)",
            "Takeoff Velocity (m/s)",
            "Vastus medialis Imbalance (%)",
        ],
        "mean": [12.67134386, 38.5789678, 2.593533839, 12.2648072],
        "std": [9.93246829, 4.870799856, 0.249450561, 8.423184381],
    }
)

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

# SJ normative values
sj_normative_values = pd.DataFrame(
    {
        "parameter": [
            "muscle_biceps_femoris_balance_%",
            "elevation_cm",
            "takeoff_velocity_m/s",
            "muscle_vastus_medialis_balance_%",
        ],
        "mean": [15.77771124, 35.92370418, 2.651046467, 14.59112967],
        "std": [9.097038503, 5.990352874, 0.264715508, 15.29637735],
    }
)
