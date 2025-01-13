import numpy as np


def get_model_9_params():
    """
    Returns all the parameters related to model 9 (GangFeng 100Ah LFP).
    """
    return {
        "Q": 0.1,  # State transition matrix 1x1
        "H": 1,  # Observation matrix 1x1
        "R": 4.71290239e9,  # Process noise covariance 1x1
        "x3": np.array(
            [
                0.0,
                0.03,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                0.98,
                1.0,
            ]
        ),
        "y3": np.array(
            [
                2.9076,
                3.1493,
                3.20125,
                3.21665,
                3.2408,
                3.26405,
                3.28,
                3.29235,
                3.297,
                3.29795,
                3.29905,
                3.3002,
                3.3019,
                3.3051,
                3.33085,
                3.3362,
                3.3364,
                3.33665,
                3.33665,
                3.33625,
                3.33575,
                3.33505,
                3.4006,
            ]
        ),
        "capacity": 100,
        "num_cells_parallel": 1,
        "num_cells_series": 16,
    }
