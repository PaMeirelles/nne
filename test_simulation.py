import unittest

from physics import Physics
from simulation import Action, BulletCreated, BulletHitPlayer, GameState, Simulation
from state import Coord, Direction, PlayerStartingState, State


class SimulationTests(unittest.TestCase):
    def make_state(self, physics, p1=(1.0, 0.5), p2=(4.0, 0.5)):
        return State(
            physics,
            (
                PlayerStartingState(Coord(*p1), Direction.EAST),
                PlayerStartingState(Coord(*p2), Direction.WEST),
            ),
        )

    def test_stationary_shooter_eventually_kills_target(self):
        physics = Physics(5, 1, player_speed=0.25, bullet_size=0.2, bullet_speed_ratio=4, match_duration=20)
        state = self.make_state(physics)
        sim = Simulation(state, physics, (lambda _: Action(shoot=True), lambda _: Action()))

        self.assertEqual(sim.run(), GameState.P1)
        self.assertTrue(any(isinstance(record.event, BulletCreated) for record in sim.events))
        self.assertTrue(any(isinstance(record.event, BulletHitPlayer) for record in sim.events))

    def test_simultaneous_shots_can_kill_both(self):
        physics = Physics(3, 1, player_speed=0.25, bullet_size=0.2, bullet_speed_ratio=8, match_duration=10)
        state = self.make_state(physics, p1=(0.75, 0.5), p2=(2.25, 0.5))
        shoot = lambda _: Action(shoot=True)
        sim = Simulation(state, physics, (shoot, shoot))

        self.assertEqual(sim.run(), GameState.SIMUL_KILLED)

    def test_timeout(self):
        physics = Physics(5, 1, player_speed=0.25, bullet_size=0.2, bullet_speed_ratio=4, match_duration=3)
        state = self.make_state(physics)
        sim = Simulation(state, physics, (lambda _: Action(), lambda _: Action()))

        self.assertEqual(sim.run(), GameState.TIMEOUT)
        self.assertEqual(sim.tick_counter, 3)
        self.assertEqual(len(sim.states), 4)  # initial + three ticks

    def test_saved_states_are_snapshots(self):
        physics = Physics(5, 1, player_speed=0.25, bullet_size=0.2, bullet_speed_ratio=4, match_duration=3)
        state = self.make_state(physics)
        move = lambda _: Action(move=Direction.EAST)
        sim = Simulation(state, physics, (move, lambda _: Action()))

        initial_x = sim.states[0].players[0].coord.x
        sim.advance_tick()
        self.assertEqual(sim.states[0].players[0].coord.x, initial_x)
        self.assertGreater(sim.states[1].players[0].coord.x, initial_x)

    def test_training_mode_does_not_record_replay(self):
        physics = Physics(5, 1, player_speed=0.25, bullet_size=0.2, bullet_speed_ratio=4, match_duration=3)
        state = self.make_state(physics)
        sim = Simulation(
            state,
            physics,
            (lambda _: Action(), lambda _: Action()),
            record_replay=False,
        )

        self.assertEqual(sim.run(), GameState.TIMEOUT)
        self.assertEqual(sim.states, [])
        self.assertEqual(sim.events, [])


if __name__ == '__main__':
    unittest.main()