import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from physics import Physics


class Direction(Enum):
    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3

@dataclass
class Coord:
    x: float
    y: float

def distance(coord1: Coord, coord2: Coord) -> float:
    return math.sqrt((coord1.x - coord2.x) ** 2 + (coord1.y - coord2.y) ** 2)

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

def player_intercept(p1_coord: Coord, p2_coord: Coord):
    return distance(p1_coord, p2_coord) < 2

class State:
    def __init__(self, physics: Physics, players: tuple[PlayerStartingState, PlayerStartingState]):
        # Player radius is .5 by default
        for player in players:
            if player.coord.x < 0.5 or player.coord.y < 0.5 or\
                    player.coord.x > (physics.board_x - 0.5) or player.coord.y > (physics.board_y - 0.5):
                raise Exception("Invalid starting state")
        if player_intercept(players[0].coord, players[1].coord):
            raise Exception("Invalid starting state")
        self.players = [Player(p.coord, p.facing, None) for p in players]
        self.bullets = [None, None]