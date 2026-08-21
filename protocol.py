from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from state import Coord, Direction, State


class GameState(Enum):
    ONGOING = 0
    P1 = 1
    P2 = 2
    TIMEOUT = 3
    SIMUL_KILLED = 4


@dataclass(frozen=True, slots=True)
class Action:
    move: bool = False
    face: Optional[Direction] = None
    shoot: bool = False


type DecisionMaker = Callable[[State], Action]


@dataclass(frozen=True, slots=True)
class GameStart:
    pass


@dataclass(frozen=True, slots=True)
class BulletCreated:
    owner: int
    coord: Coord


@dataclass(frozen=True, slots=True)
class BulletHitPlayer:
    owner: int
    target: int
    coord: Coord


@dataclass(frozen=True, slots=True)
class BulletHitWall:
    owner: int
    coord: Coord


type Event = GameStart | BulletCreated | BulletHitPlayer | BulletHitWall


@dataclass(frozen=True, slots=True)
class EventRecord:
    tick: int
    event: Event


@dataclass(frozen=True, slots=True)
class ActionRecord:
    tick: int
    actions: tuple[Action, Action]
