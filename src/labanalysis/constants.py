"""module containing useful constant values"""

# acceleration of gravity
G = 9.80665  # m/s**2

# constants for the gait detection algorithms
DEFAULT_MINIMUM_CONTACT_GRF_N = 100
DEFAULT_MINIMUM_HEIGHT_PERCENTAGE = 0.05

# constants for the jumps detection
MINIMUM_CONTACT_FORCE_N = 50
MINIMUM_FLIGHT_TIME_S = 0.1

# constants for the strength tests
MINIMUM_ISOMETRIC_DISPLACEMENT_M = 0.05

RANK_3COLORS = {
    "Normal": "#3DAE4F",
    "Fair": "#F8E262",
    "Poor": "#F28A2E",
}
RANK_4COLORS = {
    "Good": "#3FA7F5",
    "Normal": "#3DAE4F",
    "Fair": "#F8E262",
    "Poor": "#F28A2E",
}
RANK_5COLORS = {
    "Excellent": "#6A42F4",
    "Good": "#3FA7F5",
    "Normal": "#3DAE4F",
    "Fair": "#F8E262",
    "Poor": "#F28A2E",
}
SIDE_COLORS = {
    "bilateral": "#B2126F",
    "left": "#1664A8",
    "right": "#B9372A",
}
SIDE_PATTERNS = {
    "bilateral": "/",
    "left": "x",
    "right": "+",
}
