"""2x2 grid (4 signalized intersections) for the multi-intersection extension (Step 8).

* Demand: 8 fringe sources; vehicles go mostly straight through the grid (60%) and otherwise to
  any other exit (except a U-turn).  E-W is the main road, N-S the side road.
* GridVecEnv: an SB3 VecEnv in which every intersection is one "sub-environment".  A single PPO
  policy acting on it = parameter-shared independent PPO (IPPO, Schroeder de Witt et al. 2020;
  parameter sharing as in Yu et al. 2022).
* Each intersection gets the same noisy sensing, max-green enforcement and max-wait cost as the
  single-intersection env, so all single-intersection machinery (Lagrangian PPO, baselines,
  tripinfo metrics) carries over.
"""
from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

os.environ.setdefault("LIBSUMO_AS_TRACI", "1")
import sumo_rl  # noqa: E402

from .metrics import episode_metrics, read_tripinfo  # noqa: E402
from .rewards import resolve_reward  # noqa: E402
from .scenarios import Period  # noqa: E402
from .sensing import make_observation_class  # noqa: E402

GRID_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nets", "grid2x2")
GRID_NET = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "2x2grid", "2x2.net.xml")

# fringe source edge -> (approach it comes from, straight-through exit edge)
SOURCES = {
    "-h11": ("W", "-h13"), "-h21": ("W", "-h23"),   # eastbound main road
    "h13": ("E", "h11"), "h23": ("E", "h21"),       # westbound main road
    "-v11": ("N", "-v13"), "-v21": ("N", "-v23"),   # southbound side road
    "v13": ("S", "v11"), "v23": ("S", "v21"),       # northbound side road
}
SINKS = [s for _, s in SOURCES.values()]
REVERSE = {"-h11": "h11", "-h21": "h21", "h13": "-h13", "h23": "-h23",
           "-v11": "v11", "-v21": "v21", "v13": "-v13", "v23": "-v23"}

GRID_SCENARIOS: Dict[str, List[Period]] = {  # veh/h per *source edge*, keyed by approach
    "balanced": [Period(0, 3600, {"N": 400, "S": 400, "E": 400, "W": 400})],
    "asym": [Period(0, 3600, {"N": 150, "S": 150, "E": 600, "W": 600})],
    "surge": [Period(0, 1200, {"N": 150, "S": 150, "E": 600, "W": 600}),
              Period(1200, 2400, {"N": 550, "S": 550, "E": 400, "W": 400}),
              Period(2400, 3600, {"N": 150, "S": 150, "E": 600, "W": 600})],
}
THROUGH = 0.6


def write_grid_routes(periods: List[Period], path: str) -> str:
    lines = ["<routes>", '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="13.89"/>']
    flows = []
    for k, p in enumerate(periods):
        for src, (app, straight) in SOURCES.items():
            others = [s for s in SINKS if s not in (straight, REVERSE[src])]
            split = {straight: THROUGH, **{s: (1 - THROUGH) / len(others) for s in others}}
            for dst, frac in split.items():
                prob = p.vph[app] * frac / 3600.0
                fid = f"f{k}_{app}_{src}TO{dst}"  # "<k>_<approach>_<movement>" -> metrics read the approach
                flows.append((p.begin, f'    <flow id="{fid}" type="car" from="{src}" to="{dst}" begin="{p.begin}" '
                                       f'end="{p.end}" probability="{prob:.6f}" departSpeed="max" departLane="best"/>'))
    lines += [f for _, f in sorted(flows, key=lambda x: x[0])]
    lines.append("</routes>")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def grid_route_file(scenario: str) -> str:
    path = os.path.join(GRID_DIR, f"{scenario}.rou.xml")
    if not os.path.exists(path):
        write_grid_routes(GRID_SCENARIOS[scenario], path)
    return path


def make_grid_sumo(cfg: dict, seed: int, tripinfo_path: Optional[str] = None):
    e, s = cfg["env"], cfg["sensing"]
    tripinfo_path = tripinfo_path or os.path.join(tempfile.mkdtemp(prefix="grid_"), "tripinfo.xml")
    obs_cls = make_observation_class(s["noise_std"], s["dropout_p"], s["dropout_mode"], seed=seed + 10_000,
                                     dropout_p_max=s.get("dropout_p_max", 0.0),
                                     noise_std_max=s.get("noise_std_max", 0.0))
    env = sumo_rl.SumoEnvironment(
        net_file=GRID_NET, route_file=grid_route_file(e["scenario"]), num_seconds=e["num_seconds"],
        delta_time=e["delta_time"], yellow_time=e["yellow_time"], min_green=e["min_green"],
        max_green=e["max_green"], reward_fn=resolve_reward(e), observation_class=obs_cls,
        single_agent=False, sumo_seed=seed, sumo_warnings=False, add_per_agent_info=False,
        additional_sumo_cmd=f"--tripinfo-output {tripinfo_path} --tripinfo-output.write-unfinished true "
                            f"--tripinfo-output.write-undeparted true",
    )
    env.tripinfo_path = tripinfo_path
    return env


class GridVecEnv(VecEnv):
    """All intersections of one SUMO simulation as a VecEnv (one sub-env per intersection)."""

    def __init__(self, cfg: dict, seed: int = 0, lagrange_state=None):
        self.cfg, self.seed_, self.state = cfg, seed, lagrange_state
        self.sumo = make_grid_sumo(cfg, seed)
        self.ts_ids = sorted(self.sumo.ts_ids)
        self.max_green = cfg["env"]["max_green"]
        self.threshold = cfg["cost"]["threshold"]
        self.episode = 0
        self.render_mode = None
        ts0 = self.sumo.traffic_signals[self.ts_ids[0]]
        super().__init__(len(self.ts_ids), ts0.observation_space, gym.spaces.Discrete(ts0.num_green_phases))
        self._actions = None

    # -- helpers ------------------------------------------------------------------
    def _stack(self, obs: dict) -> np.ndarray:
        return np.stack([obs[t] for t in self.ts_ids]).astype(np.float32)

    def _max_wait(self, ts) -> float:
        waits = [ts.sumo.vehicle.getAccumulatedWaitingTime(v)
                 for lane in ts.lanes for v in ts.sumo.lane.getLastStepVehicleIDs(lane)]
        return float(max(waits)) if waits else 0.0

    def enforce_max_green(self, actions: dict) -> dict:
        for t, a in actions.items():
            ts = self.sumo.traffic_signals[t]
            if int(a) == ts.green_phase and ts.time_since_last_phase_change - ts.yellow_time >= self.max_green:
                actions[t] = (ts.green_phase + 1) % ts.num_green_phases
        return actions

    # -- VecEnv API -----------------------------------------------------------------
    def reset(self):
        obs = self.sumo.reset(seed=self.seed_ + self.episode)
        self.episode += 1
        return self._stack(obs)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        acts = self.enforce_max_green({t: int(a) for t, a in zip(self.ts_ids, self._actions)})
        obs, rew, dones, _ = self.sumo.step(acts)
        obs_arr = self._stack(obs)
        lam = self.state.lam if self.state is not None else 0.0
        costs = [float(self._max_wait(self.sumo.traffic_signals[t]) > self.threshold) for t in self.ts_ids]
        raw = np.array([rew[t] for t in self.ts_ids], dtype=np.float32)
        infos = [{"cost": c, "raw_reward": float(r), "lambda": lam} for c, r in zip(costs, raw)]
        done = bool(dones["__all__"])
        if done:
            for i, info in enumerate(infos):
                info["terminal_observation"] = obs_arr[i]
                info["TimeLimit.truncated"] = True
            obs_arr = self.reset()
        return obs_arr, raw - lam * np.array(costs, dtype=np.float32), np.full(self.num_envs, done), infos

    def close(self):
        self.sumo.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self, method_name)(*args, **kwargs)] * len(self._get_indices(indices))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))

    def seed(self, seed=None):
        if seed is not None:
            self.seed_ = seed
        return [self.seed_] * self.num_envs


def run_grid_episode(cfg: dict, policy, seed: int) -> dict:
    """Evaluate ``policy(obs_array, vecenv) -> actions`` for one episode; returns network metrics."""
    venv = GridVecEnv(cfg, seed=seed)
    venv.seed_ = seed
    obs = venv.sumo.reset(seed=seed)
    obs = venv._stack(obs)
    costs, done = [], False
    while not done:
        acts = venv.enforce_max_green({t: int(a) for t, a in zip(venv.ts_ids, policy(obs, venv))})
        o, _, dones, _ = venv.sumo.step(acts)
        obs = venv._stack(o)
        costs.append(np.mean([venv._max_wait(venv.sumo.traffic_signals[t]) > venv.threshold for t in venv.ts_ids]))
        done = bool(dones["__all__"])
    venv.sumo.close()
    base = venv.sumo
    m = episode_metrics(read_tripinfo(base.tripinfo_path), horizon=base.sim_max_time - base.begin_time)
    m["mean_cost"] = float(np.mean(costs))
    return m
