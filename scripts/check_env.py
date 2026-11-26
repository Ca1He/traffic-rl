"""Step-3 sanity check: random and a naive cyclic policy on each scenario."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from envs import load_config, make_env, run_episode


def cyclic(green=(30, 10, 30, 10)):
    """Naive fixed cycle; holds each phase for its green time (decisions every delta_time)."""
    state = {"t": 0}
    def policy(obs, env):
        ts = env.unwrapped.traffic_signals[env.unwrapped.ts_ids[0]]
        g = ts.time_since_last_phase_change - ts.yellow_time
        return (ts.green_phase + 1) % 4 if g >= green[ts.green_phase] else ts.green_phase
    return policy

rows = []
for scen in ["balanced", "asym", "surge"]:
    cfg = load_config(overrides={"env": {"scenario": scen}})
    for name, pol in [("random", lambda o, e: e.action_space.sample()), ("cyclic", cyclic())]:
        env = make_env(cfg, seed=0)
        t0 = time.time()
        m = run_episode(env, pol, seed=0)
        m.update(scenario=scen, policy=name, wall=round(time.time() - t0, 1))
        rows.append(m)
cols = ["scenario", "policy", "n_vehicles", "throughput_vph", "unfinished", "undeparted", "mean_delay", "p95_delay",
        "max_delay", "jain_approach", "delay_N", "delay_E", "mean_cost", "forced_switches", "wall"]
pd.set_option("display.width", 250)
print(pd.DataFrame(rows)[cols].round(2).to_string(index=False))
