import unittest

from physics import Physics
from simulation import GameState, Simulation
from protocol import BulletCreated, BulletHitPlayer, Action
from state import Coord, Direction, PlayerStartingState, State


def make_state(physics, p1=(1.0, 0.5), p2=(4.0, 0.5)):
    return State(
        physics,
        (
            PlayerStartingState(Coord(*p1), Direction.EAST),
            PlayerStartingState(Coord(*p2), Direction.WEST),
        ),
    )


class SimulationTests(unittest.TestCase):
    def test_stationary_shooter_eventually_kills_target(self):
        physics = Physics(5, 1, player_speed=0.25, bullet_size=0.2, bullet_speed_ratio=4, match_duration=20)
        state = make_state(physics)
        sim = Simulation(state, physics, (lambda _: Action(shoot=True), lambda _: Action()))

        self.assertEqual(sim.run(), GameState.P1)
        self.assertTrue(any(isinstance(record.event, BulletCreated) for record in sim.events))
        self.assertTrue(any(isinstance(record.event, BulletHitPlayer) for record in sim.events))

    def test_simultaneous_shots_can_kill_both(self):
        physics = Physics(3, 1, player_speed=0.25, bullet_size=0.2, bullet_speed_ratio=8, match_duration=10)
        state = make_state(physics, p1=(0.75, 0.5), p2=(2.25, 0.5))
        shoot = lambda _: Action(shoot=True)
        sim = Simulation(state, physics, (shoot, shoot))

        self.assertEqual(sim.run(), GameState.SIMUL_KILLED)

    def test_timeout(self):
        physics = Physics(5, 1, player_speed=0.25, bullet_size=0.2, bullet_speed_ratio=4, match_duration=3)
        state = make_state(physics)
        sim = Simulation(state, physics, (lambda _: Action(), lambda _: Action()))

        self.assertEqual(sim.run(), GameState.TIMEOUT)
        self.assertEqual(sim.tick_counter, 3)
        self.assertEqual(len(sim.states), 4)  # initial + three ticks

    def test_saved_states_are_snapshots(self):
        physics = Physics(5, 1, player_speed=0.25, bullet_size=0.2, bullet_speed_ratio=4, match_duration=3)
        state = make_state(physics)
        move = lambda _: Action(face=Direction.EAST, move=True)
        sim = Simulation(state, physics, (move, lambda _: Action()))

        initial_x = sim.states[0].players[0].coord.x
        sim.advance_tick()
        self.assertEqual(sim.states[0].players[0].coord.x, initial_x)
        self.assertGreater(sim.states[1].players[0].coord.x, initial_x)

    def test_training_mode_does_not_record_replay(self):
        physics = Physics(5, 1, player_speed=0.25, bullet_size=0.2, bullet_speed_ratio=4, match_duration=3)
        state = make_state(physics)
        sim = Simulation(
            state,
            physics,
            (lambda _: Action(), lambda _: Action()),
            record_replay=False,
        )

        self.assertEqual(sim.run(), GameState.TIMEOUT)
        self.assertEqual(sim.states, [])
        self.assertEqual(sim.events, [])


class PolicyTests(unittest.TestCase):
    def test_minimum_movement_shooter_chooses_shorter_alignment_axis(self):
        from policies import minimum_movement_shooter

        physics = Physics(5, 5, player_speed=0.25, bullet_size=0.2, bullet_speed_ratio=4, match_duration=50)
        state = State(
            physics,
            (
                PlayerStartingState(Coord(1.0, 1.0), Direction.EAST),
                PlayerStartingState(Coord(4.0, 3.0), Direction.WEST),
            ),
        )
        policy = minimum_movement_shooter(0, physics)

        self.assertEqual(policy(state), Action(face=Direction.NORTH, move=True))

    def test_minimum_movement_shooter_turns_and_shoots_when_aligned(self):
        from policies import minimum_movement_shooter

        physics = Physics(5, 5, player_speed=0.25, bullet_size=0.2, bullet_speed_ratio=4, match_duration=50)
        state = State(
            physics,
            (
                PlayerStartingState(Coord(1.0, 1.0), Direction.NORTH),
                PlayerStartingState(Coord(4.0, 1.0), Direction.WEST),
            ),
        )
        policy = minimum_movement_shooter(0, physics)

        self.assertEqual(policy(state), Action(face=Direction.EAST, shoot=True))

    def test_minimum_movement_shooter_kills_stationary_target(self):
        from policies import minimum_movement_shooter

        physics = Physics(5, 5, player_speed=0.25, bullet_size=0.2, bullet_speed_ratio=4, match_duration=50)
        state = State(
            physics,
            (
                PlayerStartingState(Coord(1.0, 1.0), Direction.EAST),
                PlayerStartingState(Coord(4.0, 3.0), Direction.WEST),
            ),
        )
        sim = Simulation(
            state,
            physics,
            (minimum_movement_shooter(0, physics), lambda _: Action()),
        )

        self.assertEqual(sim.run(), GameState.P1)


if __name__ == '__main__':
    unittest.main()