# src/tracking/shuttle_tracker.py
import numpy as np
from filterpy.kalman import KalmanFilter
from collections import deque

class ShuttleTracker:
    def __init__(self, trail_length=30):
        self.kf = self._build_kalman()
        self.trail = deque(maxlen=trail_length)  # stores (x, y) history
        self.initialized = False
        self.lost_count = 0          # frames since last real detection
        self.MAX_LOST = 15           # predict for max this many frames

    def _build_kalman(self):
        """
        State vector: [x, y, vx, vy]
        - x, y   : position
        - vx, vy : velocity
        """
        kf = KalmanFilter(dim_x=4, dim_z=2)

        dt = 1.0  # time step (1 frame)

        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)

        # Measurement matrix (we only observe x, y)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # Measurement noise (how much we trust the detector)
        kf.R *= 5.0

        # Process noise (how much we expect the shuttle to deviate)
        kf.Q *= 0.1

        # Initial covariance
        kf.P *= 100.0

        return kf

    def update(self, detection=None):
        """
        Call every frame.
        - detection: (cx, cy) if shuttle detected, else None
        Returns: (x, y) estimated position or None if too many frames lost
        """
        if detection is not None:
            cx, cy = detection

            if not self.initialized:
                # First detection — initialize state
                self.kf.x = np.array([[cx], [cy], [0.], [0.]])
                self.initialized = True
                self.lost_count = 0
            else:
                self.kf.predict()
                self.kf.update(np.array([[cx], [cy]]))
                self.lost_count = 0

        else:
            # No detection — predict only
            if self.initialized:
                self.kf.predict()
                self.lost_count += 1

        if not self.initialized or self.lost_count > self.MAX_LOST:
            return None

        x = int(self.kf.x[0])
        y = int(self.kf.x[1])
        self.trail.append((x, y))
        return (x, y)

    def get_trail(self):
        return list(self.trail)

    def reset(self):
        self.kf = self._build_kalman()
        self.trail.clear()
        self.initialized = False
        self.lost_count = 0

        # Add court bounds to ShuttleTracker.__init__
def __init__(self, trail_length=30, court_bounds=None):
    ...
    self.court_bounds = court_bounds  # (x_min, y_min, x_max, y_max)

# Update the return value in update()
def update(self, detection=None):
    ...
    x = int(self.kf.x[0])
    y = int(self.kf.x[1])

    # Clamp to court bounds if provided
    if self.court_bounds:
        x_min, y_min, x_max, y_max = self.court_bounds
        x = max(x_min, min(x_max, x))
        y = max(y_min, min(y_max, y))

    self.trail.append((x, y))
    return (x, y)