"""module containing useful constant values"""

# acceleration of gravity
G = 9.80665  # m/s**2

# constants for the gait detection algorithms
DEFAULT_MINIMUM_CONTACT_GRF_N = 100
DEFAULT_MINIMUM_HEIGHT_PERCENTAGE = 0.05

# constants for the jumps detection
MINIMUM_CONTACT_FORCE_N = 30
MINIMUM_FLIGHT_TIME_S = 0.1

# constants for the strength tests
MINIMUM_ISOMETRIC_DISPLACEMENT_M = 0.05

RANK_4COLORS = {
    "Good": "#54A3E7",
    "Normal": "#8EE09D",
    "Fair": "#F8E262",
    "Poor": "#EE9547",
}
RANK_5COLORS = {
    "Excellent": "#54A3E7",
    "Good": "#27BD6B",
    "Normal": "#8EE09D",
    "Fair": "#F8E262",
    "Poor": "#EE9547",
}
SIDE_COLORS = {
    "bilateral": "#8E0DB6",
    "left": "#1773C4",
    "right": "#C74C1C",
}
SIDE_PATTERNS = {
    "bilateral": "/",
    "left": "x",
    "right": "+",
}
