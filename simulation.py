from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from collections.abc import Callable

from mechanics import resolve_players_move, _spawn_bullet, _advance_bullet
from physics import Physics
from protocol import GameStart, BulletCreated, BulletHitPlayer, EventRecord, ActionRecord
from state import (
    Coord,
    Direction,
    State,
)


class GameState(Enum):
    ONGOING = 0
    P1 = 1
    P2 = 2
    TIMEOUT = 3
    SIMUL_KILLED = 4


@dataclass(frozen=True, slots=True)
class Action:
    move: Optional[Direction] = None
    shoot: bool = False


type DecisionMaker = Callable[[State], Action]


class Simulation:
    def __init__(
        self,
        initial_state: State,
        physics: Physics,
        decision_functions: tuple[DecisionMaker, DecisionMaker],
        record_replay: bool = True,
    ):
        self.current_state = deepcopy(initial_state)
        self.physics = physics
        self.record_replay = record_replay
        # When recording, states[n] is the complete state after n ticks; states[0] is initial.
        self.states: list[State] = [deepcopy(self.current_state)] if record_replay else []
        self.events: list[EventRecord] = (
            [EventRecord(0, GameStart())] if record_replay else []
        )
        self.actions: list[ActionRecord] = []
        self.tick_counter = 0
        self.decision_functions = decision_functions
        self.game_state = GameState.ONGOING

    def advance_tick(self) -> GameState:
        if self.game_state != GameState.ONGOING:
            return self.game_state

        actions = (
            self.decision_functions[0](self.current_state),
            self.decision_functions[1](self.current_state),
        )

        self.actions.append(ActionRecord(self.tick_counter, actions))

        new_state = deepcopy(self.current_state) if self.record_replay else self.current_state
        next_tick = self.tick_counter + 1

        player_paths = resolve_players_move(new_state, actions, self.physics)

        bullet_radius = self.physics.bullet_size / 2
        for owner, action in enumerate(actions):
            player = new_state.players[owner]
            if action.shoot and player.bullet is None:
                player.bullet = _spawn_bullet(player_paths[owner][0], player.facing, bullet_radius)
                if self.record_replay:
                    self.events.append(
                        EventRecord(
                            next_tick,
                            BulletCreated(
                                owner=owner,
                                coord=Coord(player.bullet.coord.x, player.bullet.coord.y),
                            ),
                        )
                    )

        # Resolve both bullets before deciding the winner, so simultaneous kills are possible.
        bullet_results = [
            _advance_bullet(new_state, owner, self.physics, player_paths[1 - owner])
            for owner in range(2)
        ]

        killed = [False, False]
        for owner, (next_bullet, event) in enumerate(bullet_results):
            new_state.players[owner].bullet = next_bullet
            if event is not None:
                if self.record_replay:
                    self.events.append(EventRecord(next_tick, event))
                if isinstance(event, BulletHitPlayer):
                    killed[event.target] = True

        if killed[0] and killed[1]:
            self.game_state = GameState.SIMUL_KILLED
        elif killed[0]:
            self.game_state = GameState.P2
        elif killed[1]:
            self.game_state = GameState.P1

        self.tick_counter = next_tick
        if self.game_state == GameState.ONGOING and self.tick_counter >= self.physics.match_duration:
            self.game_state = GameState.TIMEOUT

        self.current_state = new_state
        if self.record_replay:
            self.states.append(deepcopy(new_state))
        return self.game_state

    def run(self) -> GameState:
        while self.game_state == GameState.ONGOING:
            self.advance_tick()
        return self.game_state