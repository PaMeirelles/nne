from __future__ import annotations

from physics import PLAYER_RADIUS, Physics
from protocol import Action, DecisionMaker
from state import Direction, State


def _towards(delta: float, negative: Direction, positive: Direction) -> Direction:
    return positive if delta > 0 else negative


def minimum_movement_shooter(player_index: int, physics: Physics) -> DecisionMaker:
    if player_index not in (0, 1):
        raise ValueError("player_index must be 0 or 1")

    hit_radius = PLAYER_RADIUS + physics.bullet_size / 2

    def decide(state: State) -> Action:
        me = state.players[player_index]
        enemy = state.players[1 - player_index]

        dx = enemy.coord.x - me.coord.x
        dy = enemy.coord.y - me.coord.y

        if abs(dy) <= hit_radius:
            facing = _towards(dx, Direction.WEST, Direction.EAST)
            return Action(shoot=facing)

        if abs(dx) <= hit_radius:
            facing = _towards(dy, Direction.SOUTH, Direction.NORTH)
            return Action(shoot=facing)

        vertical_displacement = abs(dx) - hit_radius   # move E/W to shoot N/S
        horizontal_displacement = abs(dy) - hit_radius  # move N/S to shoot E/W

        if vertical_displacement < horizontal_displacement:
            facing = _towards(dx, Direction.WEST, Direction.EAST)
        else:
            facing = _towards(dy, Direction.SOUTH, Direction.NORTH)

        return Action(move=facing)

    return decide
