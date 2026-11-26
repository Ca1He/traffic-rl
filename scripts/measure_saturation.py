"""Measure the saturation flow (veh/h/lane) of through and left lanes in SUMO, for Webster timing.

Oversaturated demand on the N approach; the N-S green phase (through) or left phase is held
green and we count vehicles crossing the stop line once the queue has formed.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["LIBSUMO_AS_TRACI"] = "1"
import libsumo as traci
from envs.scenarios import NET_FILE, Period, write_routes

def measure(lane, state, demand=3000, hold=1800, warm=120):
    vph = {"N": demand, "S": 0, "E": 0, "W": 0}
    rou = write_routes([Period(0, hold, vph)], os.path.join(tempfile.mkdtemp(), "sat.rou.xml"))
    traci.start(["sumo", "-n", NET_FILE, "-r", rou, "--no-warnings", "--seed", "0"])
    traci.trafficlight.setRedYellowGreenState("t", state)
    passed, seen = 0, set()
    for t in range(hold):
        traci.simulationStep()
        traci.trafficlight.setRedYellowGreenState("t", state)
        now = set(traci.lane.getLastStepVehicleIDs(lane))
        if t >= warm:
            passed += len(seen - now)  # vehicles that left the approach lane this step
        seen = now
    traci.close()
    return passed * 3600 / (hold - warm)

print("through+right lane n_t_0: %.0f veh/h" % measure("n_t_0", "GGGrrrrrrrrr"))
print("left lane          n_t_1: %.0f veh/h" % measure("n_t_1", "GGGrrrrrrrrr", demand=5900))
