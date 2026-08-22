from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import neat

from physics import Physics
from protocol import Action, GameState, ReplayInfo
from simulation import Simulation
from state import Coord, Direction, State


CONFIG_PATH = Path(__file__).with_name("neat-config.ini")

# Output zero means "do nothing".  The rest are intentionally blunt: four
# movement choices followed by four shooting choices.
OUTPUT_ACTIONS = (
    Action(),
    *(Action(move=direction) for direction in Direction),
    *(Action(shoot=direction) for direction in Direction),
)


def idle(_: State) -> Action:
    return Action()


@dataclass(frozen=True, slots=True)
class TrainingArena:
    physics: Physics
    target: Coord
    starts: tuple[Coord, ...]

    @classmethod
    def stationary_target(cls) -> TrainingArena:
        physics = Physics(
            board_x=5,
            board_y=5,
            player_speed=0.25,
            bullet_size=0.2,
            bullet_speed_ratio=5,
            match_duration=20,
        )
        return cls(
            physics=physics,
            target=Coord(2.5, 2.5),
            starts=(
                Coord(0.75, 2.5),
                Coord(4.25, 2.5),
                Coord(2.5, 0.75),
                Coord(2.5, 4.25),
            ),
        )

    def state(self, start_index: int) -> State:
        start = self.starts[start_index]
        return State(
            self.physics,
            (Coord(start.x, start.y), Coord(self.target.x, self.target.y)),
        )


def encode_state(state: State, physics: Physics) -> tuple[float, ...]:
    """Return a small, normalized observation from player one's perspective."""
    me, target = state.players
    return (
        (target.coord.x - me.coord.x) / physics.board_x,
        (target.coord.y - me.coord.y) / physics.board_y,
        2 * me.coord.x / physics.board_x - 1,
        2 * me.coord.y / physics.board_y - 1,
        1.0 if me.bullet is not None else -1.0,
    )


def network_policy(network: object, physics: Physics):
    def decide(state: State) -> Action:
        outputs: Sequence[float] = network.activate(encode_state(state, physics))
        choice = max(range(len(OUTPUT_ACTIONS)), key=outputs.__getitem__)
        return OUTPUT_ACTIONS[choice]

    return decide


class StationaryTargetEvaluator:
    def __init__(self, arena: TrainingArena | None = None):
        self.arena = arena or TrainingArena.stationary_target()

    def evaluate_genome(self, genome: neat.DefaultGenome, config: neat.Config) -> float:
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        policy = network_policy(network, self.arena.physics)
        fitness = 0.0

        for start_index in range(len(self.arena.starts)):
            simulation = Simulation(
                self.arena.state(start_index),
                self.arena.physics,
                (policy, idle),
                record_replay=False,
            )
            result = simulation.run()
            if result == GameState.P1:
                # Each of the four directions is worth 25 points.  The small
                # speed bonus breaks ties without dominating actual hits.
                fitness += 25.0
                fitness += 1.0 - simulation.tick_counter / self.arena.physics.match_duration

        return fitness

    def __call__(self, genomes, config: neat.Config) -> None:
        for _, genome in genomes:
            genome.fitness = self.evaluate_genome(genome, config)

    def replay(self, genome: neat.DefaultGenome, config: neat.Config) -> ReplayInfo:
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        simulation = Simulation(
            self.arena.state(0),
            self.arena.physics,
            (network_policy(network, self.arena.physics), idle),
        )
        simulation.run()
        return ReplayInfo(simulation.states, simulation.actions, simulation.events)


def load_config() -> neat.Config:
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )


def train(generations: int, seed: int = 7, verbose: bool = True):
    config = load_config()
    population = neat.Population(config, seed=seed)
    statistics = neat.StatisticsReporter()
    population.add_reporter(statistics)
    if verbose:
        population.add_reporter(neat.StdOutReporter(True))

    evaluator = StationaryTargetEvaluator()
    winner = population.run(evaluator, generations)
    return winner, config, statistics, evaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NEAT against a stationary target.")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="open the pygame replay viewer for the winning genome",
    )
    args = parser.parse_args()
    if args.generations <= 0:
        parser.error("--generations must be positive")

    winner, config, _, evaluator = train(args.generations, args.seed)
    fitness = evaluator.evaluate_genome(winner, config)
    print(f"\nWinner fitness: {fitness:.2f} / 104.00")

    if args.replay:
        from replay import ReplayViewer

        ReplayViewer(evaluator.replay(winner, config), evaluator.arena.physics).run()


if __name__ == "__main__":
    main()
