import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from physics import Physics, PLAYER_DIAMETER, PLAYER_RADIUS


class Direction(Enum):
    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3


@dataclass
class Coord:
    x: float
    y: float


@dataclass
class Bullet:
    coord: Coord
    facing: Direction


@dataclass
class Player:
    coord: Coord
    facing: Direction
    bullet: Optional[Bullet]


@dataclass
class PlayerStartingState:
    coord: Coord
    facing: Direction


def distance(coord1: Coord, coord2: Coord) -> float:
    return math.hypot(coord1.x - coord2.x, coord1.y - coord2.y)


def player_intercept(p1_coord: Coord, p2_coord: Coord) -> bool:
    return distance(p1_coord, p2_coord) < PLAYER_DIAMETER


class State:
    def __init__(self, physics: Physics, players: tuple[PlayerStartingState, PlayerStartingState]):
        for player in players:
            if (
                player.coord.x < PLAYER_RADIUS
                or player.coord.y < PLAYER_RADIUS
                or player.coord.x > physics.board_x - PLAYER_RADIUS
                or player.coord.y > physics.board_y - PLAYER_RADIUS
            ):
                raise ValueError("Invalid starting state: player outside board")

        if player_intercept(players[0].coord, players[1].coord):
            raise ValueError("Invalid starting state: players overlap")

        self.players = [
            Player(Coord(p.coord.x, p.coord.y), p.facing, None)
            for p in players
        ]

    @property
    def bullets(self) -> tuple[Optional[Bullet], Optional[Bullet]]:
        """The active bullet owned by each player, if any."""
        return self.players[0].bullet, self.players[1].bullet