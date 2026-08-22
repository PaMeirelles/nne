from policies import minimum_movement_shooter
from protocol import ReplayInfo
from replay import ReplayViewer
from simulation import Simulation
from test_simulation import make_physics, make_state, idle

physics = make_physics(board_y=5, match_duration=50)
state = make_state(physics, p1=(1.0, 1.0), p2=(4.0, 3.0))
sim = Simulation(
    state,
    physics,
    (minimum_movement_shooter(0, physics), idle),
)

result = sim.run()

replay = ReplayViewer(ReplayInfo(sim.states, sim.actions, sim.events), physics)
replay.run()
