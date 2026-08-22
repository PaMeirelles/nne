import unittest

from neat_training import (
    OUTPUT_ACTIONS,
    TrainingArena,
    encode_state,
    network_policy,
    train,
)
from physics import PLAYER_DIAMETER, Physics
from policies import minimum_movement_shooter
from protocol import (
    Action,
    BulletCreated,
    BulletHitPlayer,
    BulletHitWall,
    GameStart,
    GameState,
)
from simulation import Simulation
from state import Bullet, Coord, Direction, State, distance


def make_physics(
    board_x=5,
    board_y=1,
    *,
    player_speed=0.25,
    bullet_size=0.2,
    bullet_speed_ratio=4,
    match_duration=20,
):
    return Physics(
        board_x,
        board_y,
        player_speed,
        bullet_size,
        bullet_speed_ratio,
        match_duration,
    )


def make_state(physics, p1=(1.0, 0.5), p2=(4.0, 0.5)):
    return State(physics, (Coord(*p1), Coord(*p2)))


def idle(_):
    return Action()


class SimulationTests(unittest.TestCase):
    def test_stationary_shooter_eventually_kills_target(self):
        physics = make_physics()
        sim = Simulation(
            make_state(physics),
            physics,
            (lambda _: Action(shoot=Direction.EAST), idle),
        )

        self.assertEqual(sim.run(), GameState.P1)
        self.assertTrue(any(isinstance(r.event, BulletCreated) for r in sim.events))
        self.assertTrue(any(isinstance(r.event, BulletHitPlayer) for r in sim.events))

    def test_simultaneous_opposing_shots_can_kill_both(self):
        physics = make_physics(
            board_x=3,
            player_speed=0.25,
            bullet_speed_ratio=8,
            match_duration=10,
        )
        sim = Simulation(
            make_state(physics, p1=(0.75, 0.5), p2=(2.25, 0.5)),
            physics,
            (
                lambda _: Action(shoot=Direction.EAST),
                lambda _: Action(shoot=Direction.WEST),
            ),
        )

        self.assertEqual(sim.run(), GameState.SIMUL_KILLED)
        hits = [r.event for r in sim.events if isinstance(r.event, BulletHitPlayer)]
        self.assertEqual({event.target for event in hits}, {0, 1})

    def test_timeout_records_initial_state_and_each_tick(self):
        physics = make_physics(match_duration=3)
        sim = Simulation(make_state(physics), physics, (idle, idle))

        self.assertEqual(sim.run(), GameState.TIMEOUT)
        self.assertEqual(sim.tick_counter, 3)
        self.assertEqual(len(sim.states), 4)
        self.assertEqual(sim.events, [sim.events[0]])
        self.assertIsInstance(sim.events[0].event, GameStart)
        self.assertEqual(sim.events[0].tick, 0)
        self.assertEqual([record.tick for record in sim.actions], [0, 1, 2])

    def test_saved_states_are_independent_snapshots(self):
        physics = make_physics(match_duration=3)
        sim = Simulation(
            make_state(physics),
            physics,
            (lambda _: Action(move=Direction.EAST), idle),
        )

        initial_x = sim.states[0].players[0].coord.x
        sim.advance_tick()

        self.assertEqual(sim.states[0].players[0].coord.x, initial_x)
        self.assertGreater(sim.states[1].players[0].coord.x, initial_x)
        self.assertIsNot(sim.states[0], sim.states[1])

    def test_training_mode_records_actions_but_not_replay(self):
        physics = make_physics(match_duration=3)
        sim = Simulation(
            make_state(physics),
            physics,
            (idle, idle),
            record_replay=False,
        )

        self.assertEqual(sim.run(), GameState.TIMEOUT)
        self.assertEqual(sim.states, [])
        self.assertEqual(sim.events, [])
        self.assertEqual(len(sim.actions), 3)

    def test_players_stop_at_first_contact(self):
        physics = make_physics(board_x=4, player_speed=1, match_duration=2)
        sim = Simulation(
            make_state(physics, p1=(1.0, 0.5), p2=(3.0, 0.5)),
            physics,
            (
                lambda _: Action(move=Direction.EAST),
                lambda _: Action(move=Direction.WEST),
            ),
        )

        sim.advance_tick()

        self.assertAlmostEqual(
            distance(sim.current_state.players[0].coord, sim.current_state.players[1].coord),
            PLAYER_DIAMETER,
        )
        self.assertLess(sim.current_state.players[0].coord.x, sim.current_state.players[1].coord.x)

    def test_player_movement_is_clamped_to_board(self):
        physics = make_physics(player_speed=2, match_duration=2)
        sim = Simulation(
            make_state(physics, p1=(0.5, 0.5), p2=(4.0, 0.5)),
            physics,
            (lambda _: Action(move=Direction.WEST), idle),
        )

        sim.advance_tick()

        self.assertEqual(sim.current_state.players[0].coord, Coord(0.5, 0.5))

    def test_bullet_hitting_wall_is_removed_and_recorded(self):
        physics = make_physics(match_duration=5)
        state = make_state(physics)
        state.players[0].bullet = Bullet(Coord(4.8, 0.5), Direction.EAST)
        sim = Simulation(state, physics, (idle, idle))

        sim.advance_tick()

        self.assertIsNone(sim.current_state.players[0].bullet)
        wall_hits = [r for r in sim.events if isinstance(r.event, BulletHitWall)]
        self.assertEqual(len(wall_hits), 1)
        self.assertEqual(wall_hits[0].tick, 1)
        self.assertAlmostEqual(wall_hits[0].event.coord.x, 4.9)

    def test_player_cannot_shoot_while_their_bullet_is_active(self):
        physics = make_physics(bullet_speed_ratio=1, match_duration=5)
        sim = Simulation(
            make_state(physics),
            physics,
            (lambda _: Action(shoot=Direction.EAST), idle),
        )

        sim.advance_tick()
        sim.advance_tick()

        created = [r for r in sim.events if isinstance(r.event, BulletCreated)]
        self.assertEqual(len(created), 1)

    def test_advance_tick_after_game_end_has_no_effect(self):
        physics = make_physics(board_x=2, bullet_speed_ratio=8, match_duration=5)
        sim = Simulation(
            make_state(physics, p1=(0.5, 0.5), p2=(1.5, 0.5)),
            physics,
            (lambda _: Action(shoot=Direction.EAST), idle),
        )
        self.assertEqual(sim.advance_tick(), GameState.P1)
        snapshot = (sim.tick_counter, len(sim.states), len(sim.events), len(sim.actions))

        self.assertEqual(sim.advance_tick(), GameState.P1)
        self.assertEqual(
            (sim.tick_counter, len(sim.states), len(sim.events), len(sim.actions)),
            snapshot,
        )

    def test_minimum_movement_shooter_kills_stationary_target(self):
        physics = make_physics(board_y=5, match_duration=50)
        state = make_state(physics, p1=(1.0, 1.0), p2=(4.0, 3.0))
        sim = Simulation(
            state,
            physics,
            (minimum_movement_shooter(0, physics), idle),
        )

        self.assertEqual(sim.run(), GameState.P1)

    def test_neat_observation_and_output_mapping_can_hit_stationary_target(self):
        arena = TrainingArena.stationary_target()
        state = arena.state(0)

        self.assertEqual(encode_state(state, arena.physics), (0.35, 0.0, -0.7, 0.0, -1.0))
        self.assertEqual(OUTPUT_ACTIONS[8], Action(shoot=Direction.EAST))

        class ShootEastNetwork:
            @staticmethod
            def activate(_):
                return [0.0] * 8 + [1.0]

        sim = Simulation(
            state,
            arena.physics,
            (network_policy(ShootEastNetwork(), arena.physics), idle),
        )
        self.assertEqual(sim.run(), GameState.P1)

    def test_neat_training_improves_fitness(self):
        winner, config, statistics, evaluator = train(
            generations=10, seed=7, verbose=False
        )
        best_by_generation = statistics.get_fitness_stat(max)

        self.assertGreater(best_by_generation[-1], best_by_generation[0])
        self.assertGreaterEqual(evaluator.evaluate_genome(winner, config), 103.0)


if __name__ == "__main__":
    unittest.main()
