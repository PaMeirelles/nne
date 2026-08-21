from __future__ import annotations

import math
from typing import Optional

from physics import Physics, PLAYER_RADIUS, PLAYER_DIAMETER
from protocol import BulletHitPlayer, BulletHitWall
from simulation import Action
from state import Direction, Coord, State, Bullet


def _direction_vector(direction: Direction) -> tuple[float, float]:
    match direction:
        case Direction.NORTH:
            return 0.0, 1.0
        case Direction.SOUTH:
            return 0.0, -1.0
        case Direction.WEST:
            return -1.0, 0.0
        case Direction.EAST:
            return 1.0, 0.0
    raise ValueError(f"Unknown direction: {direction}")


def _clamp_player(coord: Coord, physics: Physics) -> Coord:
    return Coord(
        min(max(coord.x, PLAYER_RADIUS), physics.board_x - PLAYER_RADIUS),
        min(max(coord.y, PLAYER_RADIUS), physics.board_y - PLAYER_RADIUS),
    )


def _lerp(start: Coord, end: Coord, t: float) -> Coord:
    return Coord(start.x + (end.x - start.x) * t, start.y + (end.y - start.y) * t)


def _first_circle_contact_time(
    a_start: Coord,
    a_end: Coord,
    b_start: Coord,
    b_end: Coord,
    min_distance: float,
) -> Optional[float]:
    """First t in [0, 1] at which two linearly-moving points are min_distance apart."""
    px = a_start.x - b_start.x
    py = a_start.y - b_start.y
    vx = (a_end.x - a_start.x) - (b_end.x - b_start.x)
    vy = (a_end.y - a_start.y) - (b_end.y - b_start.y)

    c = px * px + py * py - min_distance * min_distance
    if c <= 0:
        return 0.0

    a = vx * vx + vy * vy
    if a == 0:
        return None

    b = 2 * (px * vx + py * vy)
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None

    sqrt_d = math.sqrt(discriminant)
    t1 = (-b - sqrt_d) / (2 * a)
    t2 = (-b + sqrt_d) / (2 * a)

    candidates = [t for t in (t1, t2) if 0.0 <= t <= 1.0]
    return min(candidates) if candidates else None


def resolve_players_move(
    state: State, actions: tuple[Action, Action], physics: Physics
) -> tuple[tuple[Coord, Coord], tuple[Coord, Coord]]:
    """Resolve both players' movement simultaneously.

    Movement changes facing immediately. Players move at most ``player_speed`` per tick,
    are clamped to the board, and cannot pass through one another. The returned pairs are
    each player's start/end coordinates for this tick.
    """
    starts = [Coord(player.coord.x, player.coord.y) for player in state.players]
    targets: list[Coord] = []

    for player, action in zip(state.players, actions):
        if action.move is None:
            targets.append(Coord(player.coord.x, player.coord.y))
            continue

        player.facing = action.move
        dx, dy = _direction_vector(action.move)
        targets.append(
            _clamp_player(
                Coord(
                    player.coord.x + dx * physics.player_speed,
                    player.coord.y + dy * physics.player_speed,
                ),
                physics,
            )
        )

    collision_t = _first_circle_contact_time(
        starts[0], targets[0], starts[1], targets[1], PLAYER_DIAMETER
    )

    if collision_t is None:
        state.players[0].coord = targets[0]
        state.players[1].coord = targets[1]
    else:
        # Stop both at first contact. Subtract a tiny epsilon to avoid floating-point overlap.
        t = max(0.0, collision_t - 1e-9)
        state.players[0].coord = _lerp(starts[0], targets[0], t)
        state.players[1].coord = _lerp(starts[1], targets[1], t)

    return (
        (starts[0], Coord(state.players[0].coord.x, state.players[0].coord.y)),
        (starts[1], Coord(state.players[1].coord.x, state.players[1].coord.y)),
    )


def _spawn_bullet(player_coord: Coord, facing: Direction, bullet_radius: float) -> Bullet:
    dx, dy = _direction_vector(facing)
    separation = PLAYER_RADIUS + bullet_radius
    return Bullet(
        Coord(
            player_coord.x + dx * separation,
            player_coord.y + dy * separation,
        ),
        facing,
    )


def _first_wall_contact_time(
    start: Coord,
    end: Coord,
    radius: float,
    physics: Physics,
) -> Optional[float]:
    """First t in [0,1] at which a moving circle reaches the board boundary."""
    min_x = radius
    max_x = physics.board_x - radius
    min_y = radius
    max_y = physics.board_y - radius

    if start.x < min_x or start.x > max_x or start.y < min_y or start.y > max_y:
        return 0.0

    dx = end.x - start.x
    dy = end.y - start.y
    candidates: list[float] = []

    if dx < 0 and end.x <= min_x:
        candidates.append((min_x - start.x) / dx)
    elif dx > 0 and end.x >= max_x:
        candidates.append((max_x - start.x) / dx)

    if dy < 0 and end.y <= min_y:
        candidates.append((min_y - start.y) / dy)
    elif dy > 0 and end.y >= max_y:
        candidates.append((max_y - start.y) / dy)

    candidates = [t for t in candidates if 0.0 <= t <= 1.0]
    return min(candidates) if candidates else None


def _advance_bullet(
    state: State,
    owner: int,
    physics: Physics,
    target_path: tuple[Coord, Coord],
) -> tuple[Optional[Bullet], Optional[Event]]:
    bullet = state.players[owner].bullet
    if bullet is None:
        return None, None

    target = 1 - owner
    dx, dy = _direction_vector(bullet.facing)
    start = Coord(bullet.coord.x, bullet.coord.y)
    end = Coord(
        start.x + dx * physics.bullet_speed,
        start.y + dy * physics.bullet_speed,
    )

    bullet_radius = physics.bullet_size / 2
    hit_radius = PLAYER_RADIUS + bullet_radius

    target_start, target_end = target_path
    hit_t = _first_circle_contact_time(start, end, target_start, target_end, hit_radius)
    wall_t = _first_wall_contact_time(start, end, bullet_radius, physics)

    if hit_t is not None and (wall_t is None or hit_t <= wall_t):
        coord = _lerp(start, end, hit_t)
        return None, BulletHitPlayer(owner=owner, target=target, coord=coord)

    if wall_t is not None:
        coord = _lerp(start, end, wall_t)
        coord = Coord(
            min(max(coord.x, bullet_radius), physics.board_x - bullet_radius),
            min(max(coord.y, bullet_radius), physics.board_y - bullet_radius),
        )
        return None, BulletHitWall(owner=owner, coord=coord)

    return Bullet(end, bullet.facing), None
