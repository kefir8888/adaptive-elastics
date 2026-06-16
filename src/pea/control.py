"""Deploy-time adaptive controllers for the parallel knee preload.

Extracted from go1_capacity.py and render_walk.py (which carried verbatim copies — if the gains
drift the rendered video silently disagrees with the measured capacity curve). Single source of truth.
"""
from __future__ import annotations

import numpy as np


class AdaptivePreloadController:
    """Per-leg, self-tuning constant knee preload — the project's positive-result mechanism.

    A slow outer loop around the fast RL policy: each leg keeps a ~15 s exponential moving average
    of its OWN motor knee (calf) torque and ramps a passive preload tau0 (clipped-proportional,
    bounded rate) to drive that mean toward `target` (0 = full compensation). No load sensor, no
    payload observation. `update(calf_motor_torque)` is called once per control step with the
    per-leg motor torque and returns the current per-leg tau0 vector.
    """

    def __init__(self, n_legs: int, dt: float, tau0_max: float,
                 window_s: float = 15.0, kp: float = 0.2, rate: float = 2.0, target: float = 0.0):
        self.dt = float(dt)
        self.alpha = float(np.exp(-self.dt / window_s))  # EMA decay for the window
        self.kp = float(kp)
        self.rate = float(rate)                          # max |dtau0/dt| (N*m/s)
        self.target = float(target)
        self.tmax = float(tau0_max)
        self.tau0 = np.zeros(n_legs)
        self.ema = np.zeros(n_legs)

    def update(self, calf_motor_torque) -> np.ndarray:
        cm = np.asarray(calf_motor_torque, dtype=float)
        self.ema = self.alpha * self.ema + (1.0 - self.alpha) * cm
        step = np.clip(self.kp * (self.ema - self.target), -self.rate, self.rate) * self.dt
        self.tau0 = np.clip(self.tau0 + step, 0.0, self.tmax)
        return self.tau0
