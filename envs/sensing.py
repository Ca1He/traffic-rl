"""Sensor model: lane detectors with optional Gaussian noise and detector dropout.

Every controller (RL agents *and* Max-Pressure) reads traffic state through this
layer, so robustness experiments perturb all methods identically.

Observation layout (same as sumo-rl's default, so results are comparable):
    [phase one-hot (P) | min-green satisfied (1) | in-lane density (L) | in-lane queue (L)]
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces
from sumo_rl.environment.observations import ObservationFunction


class NoisyObservation(ObservationFunction):
    """Lane-detector observation with configurable noise.

    Class attributes (set through :func:`make_observation_class`):
        noise_std:     std of additive Gaussian noise on normalized density/queue readings.
        dropout_p:     probability that a lane detector reports nothing (reads 0).
        dropout_mode:  "step"    -> failures redrawn at every decision step (transient faults);
                       "episode" -> a fixed set of detectors fails for the whole episode.
        rng:           shared numpy Generator (seeded for reproducibility).
    """

    noise_std: float = 0.0
    dropout_p: float = 0.0
    dropout_mode: str = "step"
    dropout_p_max: float = 0.0   # >0: draw dropout_p ~ U[0, max] every episode (domain randomization)
    noise_std_max: float = 0.0   # >0: draw noise_std ~ U[0, max] every episode
    rng: np.random.Generator = np.random.default_rng(0)

    def __init__(self, ts):
        super().__init__(ts)
        # NOTE: sumo-rl creates the observation function *before* ts.lanes exists,
        # so anything lane-dependent is initialised lazily on the first call.
        self._episode_mask_in: Optional[np.ndarray] = None
        self._episode_mask_out: Optional[np.ndarray] = None
        self.out_lanes = None
        self.last = {}  # last sensed (possibly corrupted) readings, read by Max-Pressure
        # sumo-rl builds a new observation function at every reset -> one draw per episode.
        # Instance attributes shadow the class-level settings.
        if self.dropout_p_max > 0:
            self.dropout_p = float(self.rng.uniform(0.0, self.dropout_p_max))
        if self.noise_std_max > 0:
            self.noise_std = float(self.rng.uniform(0.0, self.noise_std_max))

    # ------------------------------------------------------------------
    def _out_density(self) -> np.ndarray:
        ts = self.ts
        return np.array(
            [
                min(1.0, ts.sumo.lane.getLastStepVehicleNumber(l)
                    / (ts.lanes_length[l] / (ts.MIN_GAP + ts.sumo.lane.getLastStepLength(l))))
                for l in self.out_lanes
            ],
            dtype=np.float32,
        )

    def _corrupt(self, x: np.ndarray, keep: Optional[np.ndarray]) -> np.ndarray:
        """Add Gaussian noise, then zero out failed detectors (keep == False)."""
        if self.noise_std > 0:
            x = x + self.rng.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
        if keep is not None:
            x = np.where(keep, x, 0.0)
        return np.clip(x, 0.0, 1.0).astype(np.float32)

    def _keep_masks(self, n_in: int, n_out: int):
        """Which detectors work at this step (None = no dropout)."""
        if self.dropout_p <= 0:
            return None, None
        if self.dropout_mode == "episode":
            return self._episode_mask_in, self._episode_mask_out
        return self.rng.random(n_in) >= self.dropout_p, self.rng.random(n_out) >= self.dropout_p

    def _lazy_init(self):
        self.out_lanes = sorted(self.ts.out_lanes)  # sumo-rl stores a set -> fix the order
        if self.dropout_mode == "episode" and self.dropout_p > 0:
            self._episode_mask_in = self.rng.random(len(self.ts.lanes)) >= self.dropout_p
            self._episode_mask_out = self.rng.random(len(self.out_lanes)) >= self.dropout_p

    # ------------------------------------------------------------------
    def __call__(self) -> np.ndarray:
        if self.out_lanes is None:
            self._lazy_init()
        ts = self.ts
        true_density = np.array(ts.get_lanes_density(), dtype=np.float32)
        true_queue = np.array(ts.get_lanes_queue(), dtype=np.float32)
        true_out = self._out_density()

        # One physical detector per lane: a failure zeroes both its density and queue readings.
        keep_in, keep_out = self._keep_masks(len(true_density), len(true_out))
        density = self._corrupt(true_density, keep_in)
        queue = self._corrupt(true_queue, keep_in)
        out_density = self._corrupt(true_out, keep_out)

        self.last = {
            "in_lanes": list(ts.lanes),
            "out_lanes": self.out_lanes,
            "in_density": density,
            "in_queue": queue,
            "out_density": out_density,
        }
        phase = [1.0 if ts.green_phase == i else 0.0 for i in range(ts.num_green_phases)]
        min_green = [0.0 if ts.time_since_last_phase_change < ts.min_green + ts.yellow_time else 1.0]
        return np.concatenate([np.array(phase + min_green, dtype=np.float32), density, queue])

    def observation_space(self) -> spaces.Box:
        n = self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes)
        return spaces.Box(low=np.zeros(n, dtype=np.float32), high=np.ones(n, dtype=np.float32))


def make_observation_class(noise_std: float = 0.0, dropout_p: float = 0.0,
                           dropout_mode: str = "step", seed: int = 0,
                           dropout_p_max: float = 0.0, noise_std_max: float = 0.0) -> type:
    """Create a NoisyObservation subclass with the given sensor configuration."""
    if dropout_mode not in ("step", "episode"):
        raise ValueError("dropout_mode must be 'step' or 'episode'")
    return type(
        "ConfiguredNoisyObservation",
        (NoisyObservation,),
        {
            "noise_std": float(noise_std),
            "dropout_p": float(dropout_p),
            "dropout_mode": dropout_mode,
            "dropout_p_max": float(dropout_p_max or 0.0),
            "noise_std_max": float(noise_std_max or 0.0),
            "rng": np.random.default_rng(seed),
        },
    )
