from copy import copy
from dataclasses import dataclass
from typing import Optional
from collections.abc import Callable
from physics import Physics
from state import Direction, Coord, State
from enum import Enum

class GameState(Enum):
    ONGOING = 0
    P1 = 1
    P2 = 2
    TIMEOUT = 3
    SIMUL_KILLED = 4

@dataclass(frozen=True, slots=True)
class GameStart:
    pass


@dataclass(frozen=True, slots=True)
class BulletCreated:
    coord: Coord


@dataclass(frozen=True, slots=True)
class BulletHitPlayer:
    coord: Coord


@dataclass(frozen=True, slots=True)
class BulletHitWall:
    coord: Coord


type Event = (
    GameStart
    | BulletCreated
    | BulletHitPlayer
    | BulletHitWall
)

@dataclass
class Action:
    Move: Optional[Direction]
    Shoot: bool

decision_maker = Callable[[State], Action]

def resolve_players_move(state: State, directions: [tuple[Direction, Direction]], physics: Physics) -> None:
    # Check if they would hit one another (they are aligned and moving towards each other). If so, move to the middle
    # Check if they would hit a wall. If so, move then to the wall
    # Otherwise, just move
    pass


class Simulation:
    def __init__(self, initial_state: State, physics: Physics, decision_functions: tuple[decision_maker, decision_maker]):
        self.current_state = initial_state
        self.physics = physics
        self.states: list[State] = []
        self.events: list[Event] = []
        self.tick_counter = 0
        self.decision_functions = decision_functions
        self.game_state: GameState = GameState.ONGOING

    def advance_tick(self):
        if self.game_state != GameState.ONGOING: return
        action_p1 = self.decision_functions[0](self.current_state)
        action_p2 = self.decision_functions[1](self.current_state)

        new_state = copy(self.current_state)
        # We need to move bullets, which may create events and alter game state
        # Also parse shooting (whether we can, create events, etc)
        self.states.append(self.current_state)
        self.current_state = new_state

        self.tick_counter += 1
        if self.tick_counter == self.physics.match_duration:
            self.game_state = GameState.TIMEOUT



