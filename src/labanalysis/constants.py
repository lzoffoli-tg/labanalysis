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

# Marker colors for time points in strength tests
# Chosen to be well-distinguishable from SIDE_COLORS (#B2126F magenta, #1664A8 blue, #B9372A red)
MARKER_COLORS = [
    '#2ecc71',  # Emerald green
    '#f39c12',  # Orange
    '#9b59b6',  # Purple
    '#1abc9c',  # Turquoise
    '#e74c3c',  # Bright red (different from side red #B9372A)
    '#34495e',  # Dark blue-gray
    '#f1c40f',  # Yellow
    '#95a5a6',  # Light gray
    '#16a085',  # Dark cyan
    '#e67e22',  # Carrot orange
]
