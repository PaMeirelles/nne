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
    bullet: Optional[Bullet]



def distance(coord1: Coord, coord2: Coord) -> float:
    return math.hypot(coord1.x - coord2.x, coord1.y - coord2.y)


def player_intercept(p1_coord: Coord, p2_coord: Coord) -> bool:
    return distance(p1_coord, p2_coord) < PLAYER_DIAMETER


class State:
    def __init__(self, physics: Physics, starting_pos: tuple[Coord, Coord]):
        for coord in starting_pos:
            if (
                coord.x < PLAYER_RADIUS
                or coord.y < PLAYER_RADIUS
                or coord.x > physics.board_x - PLAYER_RADIUS
                or coord.y > physics.board_y - PLAYER_RADIUS
            ):
                raise ValueError("Invalid starting state: player outside board")

        if player_intercept(starting_pos[0], starting_pos[1]):
            raise ValueError("Invalid starting state: players overlap")

        self.players = [
            Player(Coord(coord.x, coord.y), None)
            for coord in starting_pos
        ]

    @property
    def bullets(self) -> tuple[Optional[Bullet], Optional[Bullet]]:
        """The active bullet owned by each player, if any."""
        return self.players[0].bullet, self.players[1].bullet