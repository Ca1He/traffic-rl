"""Step 1 smoke test: run one episode of a random policy on the single-intersection network."""
import os
import time

import numpy as np
import sumo_rl

NETS = os.path.join(os.path.dirname(sumo_rl.__file__), "nets", "2way-single-intersection")

env = sumo_rl.SumoEnvironment(
    net_file=os.path.join(NETS, "single-intersection.net.xml"),
    route_file=os.path.join(NETS, "single-intersection-vhvh.rou.xml"),
    use_gui=False,
    num_seconds=3600,
    delta_time=5,
    yellow_time=3,
    min_green=10,
    max_green=60,
    single_agent=True,
    sumo_seed=0,
)

obs, info = env.reset()
print("obs dim:", env.observation_space.shape, "| actions:", env.action_space.n)
t0 = time.time()
done, steps, total_r = False, 0, 0.0
while not done:
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    total_r += r
    steps += 1
    done = term or trunc
print(f"steps={steps}  total_reward={total_r:.1f}  wall={time.time()-t0:.1f}s")
print("final info:", {k: round(v, 2) if isinstance(v, float) else v for k, v in info.items()})
env.close()
