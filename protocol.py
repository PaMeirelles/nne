from __future__ import annotations

from dataclasses import dataclass

from simulation import Action
from state import Coord


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


@dataclass(frozen=True, slots=True)
class EventRecord:
    tick: int
    event: Event


@dataclass(frozen=True, slots=True)
class ActionRecord:
    tick: int
    actions: tuple[Action, Action]

type Event = GameStart | BulletCreated | BulletHitPlayer | BulletHitWall
