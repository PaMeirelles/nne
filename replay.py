from __future__ import annotations

from dataclasses import dataclass

import pygame

from physics import PLAYER_RADIUS, Physics
from protocol import (
    BulletCreated,
    BulletHitPlayer,
    BulletHitWall,
    EventRecord,
    GameStart,
    ReplayInfo,
)
from state import Bullet, Coord


Color = tuple[int, int, int]
AlphaColor = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    window_size: tuple[int, int] = (1100, 700)
    fps: int = 60
    ticks_per_second: float = 12.0
    start_hold_seconds: float = 0.8
    impact_hold_seconds: float = 0.45

    margin: int = 24
    sidebar_width: int = 258
    panel_gap: int = 24
    board_padding: int = 20

    title_font_size: int = 25
    body_font_size: int = 19
    overlay_title_font_size: int = 52
    overlay_body_font_size: int = 24

    background: Color = (17, 21, 29)
    surface: Color = (27, 33, 44)
    board: Color = (12, 17, 23)
    grid: Color = (35, 44, 56)
    text: Color = (232, 238, 245)
    muted: Color = (145, 157, 173)
    player_one: Color = (72, 174, 255)
    player_two: Color = (255, 108, 127)
    bullet: Color = (247, 205, 91)
    hit_player: Color = (255, 76, 104)
    hit_wall: Color = (255, 150, 76)
    overlay: AlphaColor = (5, 8, 12, 205)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.ticks_per_second <= 0:
            raise ValueError("ticks_per_second must be positive")
        if self.start_hold_seconds < 0 or self.impact_hold_seconds < 0:
            raise ValueError("hold durations cannot be negative")
        if min(self.window_size) <= 0:
            raise ValueError("window dimensions must be positive")


@dataclass(frozen=True, slots=True)
class _Fonts:
    title: pygame.font.Font
    body: pygame.font.Font
    overlay_title: pygame.font.Font
    overlay_body: pygame.font.Font


class ReplayViewer:
    def __init__(
        self,
        replay: ReplayInfo,
        physics: Physics,
        config: ReplayConfig | None = None,
    ):
        if not replay.states:
            raise ValueError("replay must contain at least one state")
        self.replay = replay
        self.physics = physics
        self.config = config or ReplayConfig()

    def run(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode(self.config.window_size)
        pygame.display.set_caption("Match replay viewer")
        clock = pygame.time.Clock()
        fonts = _Fonts(
            title=pygame.font.Font(None, self.config.title_font_size),
            body=pygame.font.Font(None, self.config.body_font_size),
            overlay_title=pygame.font.Font(None, self.config.overlay_title_font_size),
            overlay_body=pygame.font.Font(None, self.config.overlay_body_font_size),
        )

        try:
            running = True
            while running:
                running = self._play_once(screen, clock, fonts)
                if running:
                    running = self._wait_for_restart(screen, clock, fonts)
        finally:
            pygame.quit()

    def _play_once(
        self,
        screen: pygame.Surface,
        clock: pygame.time.Clock,
        fonts: _Fonts,
    ) -> bool:
        if not self._hold_start(screen, clock, fonts):
            return False

        last_tick = len(self.replay.states) - 1
        tick_duration = 1 / self.config.ticks_per_second

        for tick in range(last_tick):
            elapsed = 0.0
            while elapsed < tick_duration:
                running, delta, _ = self._pump_events(clock)
                if not running:
                    return False
                elapsed += delta
                alpha = min(1.0, elapsed / tick_duration)
                self._draw(
                    screen,
                    fonts,
                    tick=tick,
                    alpha=alpha,
                    event_tick=tick + 1,
                    event_progress=alpha,
                )
                pygame.display.flip()

            impact_events = [
                record
                for record in self._events_at(tick + 1)
                if isinstance(record.event, (BulletHitPlayer, BulletHitWall))
            ]
            if impact_events and not self._hold_impact(
                screen, clock, fonts, tick + 1
            ):
                return False

        return True

    def _hold_start(
        self,
        screen: pygame.Surface,
        clock: pygame.time.Clock,
        fonts: _Fonts,
    ) -> bool:
        elapsed = 0.0
        while elapsed < self.config.start_hold_seconds:
            running, delta, _ = self._pump_events(clock)
            if not running:
                return False
            elapsed += delta
            self._draw(screen, fonts, tick=0, alpha=0.0, event_tick=0)
            self._draw_overlay(screen, fonts, "MATCH START", "")
            pygame.display.flip()
        return True

    def _hold_impact(
        self,
        screen: pygame.Surface,
        clock: pygame.time.Clock,
        fonts: _Fonts,
        tick: int,
    ) -> bool:
        if self.config.impact_hold_seconds == 0:
            return True

        elapsed = 0.0
        while elapsed < self.config.impact_hold_seconds:
            running, delta, _ = self._pump_events(clock)
            if not running:
                return False
            elapsed += delta
            progress = min(1.0, elapsed / self.config.impact_hold_seconds)
            self._draw(
                screen,
                fonts,
                tick=tick,
                alpha=1.0,
                event_tick=tick,
                event_progress=progress,
            )
            pygame.display.flip()
        return True

    def _wait_for_restart(
        self,
        screen: pygame.Surface,
        clock: pygame.time.Clock,
        fonts: _Fonts,
    ) -> bool:
        final_tick = len(self.replay.states) - 1
        title, subtitle = self._match_result()
        restart_prompt = "PRESS ANY KEY TO REPLAY"
        if subtitle:
            restart_prompt = f"{subtitle}  ·  {restart_prompt}"

        while True:
            self._draw(screen, fonts, tick=final_tick, alpha=1.0)
            self._draw_overlay(
                screen,
                fonts,
                title,
                restart_prompt,
            )
            pygame.display.flip()

            running, _, restart = self._pump_events(clock, listen_for_restart=True)
            if not running:
                return False
            if restart:
                return True

    def _pump_events(
        self,
        clock: pygame.time.Clock,
        *,
        listen_for_restart: bool = False,
    ) -> tuple[bool, float, bool]:
        delta = clock.tick(self.config.fps) / 1000.0
        restart = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, delta, False
            if listen_for_restart and event.type == pygame.KEYDOWN:
                restart = True
        return True, delta, restart

    def _draw(
        self,
        screen: pygame.Surface,
        fonts: _Fonts,
        *,
        tick: int,
        alpha: float,
        event_tick: int | None = None,
        event_progress: float = 0.0,
    ) -> None:
        config = self.config
        screen.fill(config.background)

        title = fonts.title.render("MATCH REPLAY", True, config.text)
        screen.blit(title, (config.margin, 18))
        tick_label = fonts.body.render(
            f"tick {tick} / {len(self.replay.states) - 1}", True, config.muted
        )
        screen.blit(tick_label, (config.margin + 236, 22))

        board_panel, event_panel = self._panel_geometry(screen)
        pygame.draw.rect(screen, config.surface, board_panel, border_radius=8)
        board, scale = self._board_geometry(board_panel)
        pygame.draw.rect(screen, config.board, board)
        self._draw_grid(screen, board, scale)
        self._draw_state(screen, fonts, tick, alpha, board, scale, event_tick)

        if event_tick is not None:
            self._draw_events(
                screen, fonts, event_tick, event_progress, board, scale
            )

        displayed_tick = event_tick if event_tick is not None and alpha >= 0.5 else tick
        self._draw_event_panel(screen, fonts, event_panel, displayed_tick)

    def _panel_geometry(
        self, screen: pygame.Surface
    ) -> tuple[pygame.Rect, pygame.Rect]:
        config = self.config
        top = 68
        height = screen.get_height() - top - config.margin
        board_width = (
            screen.get_width()
            - 2 * config.margin
            - config.panel_gap
            - config.sidebar_width
        )
        board_panel = pygame.Rect(config.margin, top, board_width, height)
        event_panel = pygame.Rect(
            board_panel.right + config.panel_gap,
            top,
            config.sidebar_width,
            height,
        )
        return board_panel, event_panel

    def _board_geometry(self, panel: pygame.Rect) -> tuple[pygame.Rect, float]:
        padding = self.config.board_padding
        scale = min(
            (panel.width - 2 * padding) / self.physics.board_x,
            (panel.height - 2 * padding) / self.physics.board_y,
        )
        board = panel.inflate(-2 * padding, -2 * padding)
        board.size = (
            round(self.physics.board_x * scale),
            round(self.physics.board_y * scale),
        )
        board.center = panel.center
        return board, scale

    def _screen_coord(
        self, coord: Coord, board: pygame.Rect, scale: float
    ) -> tuple[int, int]:
        return round(board.left + coord.x * scale), round(board.bottom - coord.y * scale)

    def _draw_grid(
        self,
        screen: pygame.Surface,
        board: pygame.Rect,
        scale: float,
    ) -> None:
        for x in range(1, int(self.physics.board_x)):
            pygame.draw.line(
                screen,
                self.config.grid,
                self._screen_coord(Coord(x, 0), board, scale),
                self._screen_coord(Coord(x, self.physics.board_y), board, scale),
            )
        for y in range(1, int(self.physics.board_y)):
            pygame.draw.line(
                screen,
                self.config.grid,
                self._screen_coord(Coord(0, y), board, scale),
                self._screen_coord(Coord(self.physics.board_x, y), board, scale),
            )
        pygame.draw.rect(screen, self.config.muted, board, width=1)

    def _draw_state(
        self,
        screen: pygame.Surface,
        fonts: _Fonts,
        tick: int,
        alpha: float,
        board: pygame.Rect,
        scale: float,
        event_tick: int | None,
    ) -> None:
        previous = self.replay.states[tick]
        following = self.replay.states[min(tick + 1, len(self.replay.states) - 1)]
        transition_events = self._events_at(event_tick) if event_tick is not None else []

        for player_index in range(2):
            player_start = previous.players[player_index].coord
            player_end = following.players[player_index].coord
            player_coord = self._lerp(player_start, player_end, alpha)
            color = (
                self.config.player_one
                if player_index == 0
                else self.config.player_two
            )
            center = self._screen_coord(player_coord, board, scale)
            radius = max(7, round(PLAYER_RADIUS * scale))
            pygame.draw.circle(screen, color, center, radius)
            label = fonts.body.render(
                f"P{player_index + 1}", True, self.config.background
            )
            screen.blit(label, label.get_rect(center=center))

            bullet = self._interpolated_bullet(
                previous.players[player_index].bullet,
                following.players[player_index].bullet,
                transition_events,
                player_index,
                alpha,
            )
            if bullet is not None:
                bullet_center = self._screen_coord(bullet.coord, board, scale)
                bullet_radius = max(3, round(self.physics.bullet_size * scale / 2))
                pygame.draw.circle(
                    screen, self.config.bullet, bullet_center, bullet_radius
                )

    def _interpolated_bullet(
        self,
        previous: Bullet | None,
        following: Bullet | None,
        events: list[EventRecord],
        owner: int,
        alpha: float,
    ) -> Bullet | None:
        if previous is not None and following is not None:
            return Bullet(
                self._lerp(previous.coord, following.coord, alpha), following.facing
            )

        created = next(
            (
                record.event
                for record in events
                if isinstance(record.event, BulletCreated) and record.event.owner == owner
            ),
            None,
        )
        ended = next(
            (
                record.event
                for record in events
                if isinstance(record.event, (BulletHitPlayer, BulletHitWall))
                and record.event.owner == owner
            ),
            None,
        )

        if previous is None and following is not None:
            start = created.coord if created is not None else following.coord
            return Bullet(self._lerp(start, following.coord, alpha), following.facing)
        if previous is not None and following is None:
            end = ended.coord if ended is not None else previous.coord
            return Bullet(self._lerp(previous.coord, end, alpha), previous.facing)
        return None

    def _draw_events(
        self,
        screen: pygame.Surface,
        fonts: _Fonts,
        event_tick: int,
        progress: float,
        board: pygame.Rect,
        scale: float,
    ) -> None:
        for record in self._events_at(event_tick):
            event = record.event
            if isinstance(event, GameStart):
                continue

            center = self._screen_coord(event.coord, board, scale)
            if isinstance(event, BulletCreated):
                radius = max(6, round(18 - 8 * progress))
                pygame.draw.circle(
                    screen, self.config.bullet, center, radius, width=2
                )
                arm = max(5, round(12 * (1 - progress)))
                pygame.draw.line(
                    screen,
                    self.config.bullet,
                    (center[0] - arm, center[1]),
                    (center[0] + arm, center[1]),
                    width=2,
                )
                pygame.draw.line(
                    screen,
                    self.config.bullet,
                    (center[0], center[1] - arm),
                    (center[0], center[1] + arm),
                    width=2,
                )
            elif isinstance(event, BulletHitPlayer):
                radius = max(12, round(16 + 34 * progress))
                pygame.draw.circle(
                    screen, self.config.hit_player, center, radius, width=4
                )
                pygame.draw.circle(
                    screen,
                    self.config.bullet,
                    center,
                    max(5, round(radius * 0.25)),
                )
                label = fonts.title.render(
                    f"P{event.owner + 1} HIT P{event.target + 1}",
                    True,
                    self.config.hit_player,
                )
                screen.blit(label, label.get_rect(midbottom=(center[0], center[1] - 18)))
            elif isinstance(event, BulletHitWall):
                radius = max(8, round(10 + 20 * progress))
                pygame.draw.circle(
                    screen, self.config.hit_wall, center, radius, width=3
                )

    def _draw_event_panel(
        self,
        screen: pygame.Surface,
        fonts: _Fonts,
        panel: pygame.Rect,
        tick: int,
    ) -> None:
        pygame.draw.rect(screen, self.config.surface, panel, border_radius=8)
        heading = fonts.body.render("MATCH EVENTS", True, self.config.muted)
        screen.blit(heading, (panel.left + 18, panel.top + 20))

        events = [record for record in self.replay.events if record.tick <= tick]
        y = panel.top + 55
        if not events:
            no_events = fonts.body.render("No events", True, self.config.muted)
            screen.blit(no_events, (panel.left + 18, y))
        else:
            line_height = fonts.body.get_linesize() + 5
            available_lines = max(1, (panel.height - 90) // line_height)
            for record in events[-available_lines:]:
                text, color = self._event_text(record)
                screen.blit(
                    fonts.body.render(text, True, color),
                    (panel.left + 18, y),
                )
                y += line_height

        board_text = fonts.body.render(
            f"board {self.physics.board_x:g} × {self.physics.board_y:g}",
            True,
            self.config.muted,
        )
        screen.blit(board_text, (panel.left + 18, panel.bottom - 28))

    def _draw_overlay(
        self,
        screen: pygame.Surface,
        fonts: _Fonts,
        title: str,
        subtitle: str,
    ) -> None:
        shade = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        shade.fill(self.config.overlay)
        screen.blit(shade, (0, 0))

        title_surface = fonts.overlay_title.render(title, True, self.config.text)
        title_rect = title_surface.get_rect(center=screen.get_rect().center)
        if subtitle:
            title_rect.move_ip(0, -18)
        screen.blit(title_surface, title_rect)

        if subtitle:
            subtitle_surface = fonts.overlay_body.render(
                subtitle, True, self.config.muted
            )
            subtitle_rect = subtitle_surface.get_rect(
                midtop=(screen.get_rect().centerx, title_rect.bottom + 14)
            )
            screen.blit(subtitle_surface, subtitle_rect)

    def _match_result(self) -> tuple[str, str]:
        final_tick = len(self.replay.states) - 1
        targets = {
            record.event.target
            for record in self._events_at(final_tick)
            if isinstance(record.event, BulletHitPlayer)
        }
        if targets == {0, 1}:
            return "DOUBLE K.O.", "BOTH PLAYERS HIT"
        if 1 in targets:
            return "PLAYER 1 WINS", "PLAYER 2 HIT"
        if 0 in targets:
            return "PLAYER 2 WINS", "PLAYER 1 HIT"
        if final_tick >= self.physics.match_duration:
            return "TIMEOUT", "NO WINNER"
        return "MATCH OVER", ""

    def _events_at(self, tick: int) -> list[EventRecord]:
        return [record for record in self.replay.events if record.tick == tick]

    @staticmethod
    def _lerp(start: Coord, end: Coord, alpha: float) -> Coord:
        return Coord(
            start.x + (end.x - start.x) * alpha,
            start.y + (end.y - start.y) * alpha,
        )

    def _event_text(self, record: EventRecord) -> tuple[str, Color]:
        event = record.event
        if isinstance(event, GameStart):
            return f"{record.tick:>3}  game started", self.config.text
        if isinstance(event, BulletCreated):
            return f"{record.tick:>3}  P{event.owner + 1} fired", self.config.bullet
        if isinstance(event, BulletHitPlayer):
            return (
                f"{record.tick:>3}  P{event.owner + 1} hit P{event.target + 1}",
                self.config.hit_player,
            )
        if isinstance(event, BulletHitWall):
            return (
                f"{record.tick:>3}  P{event.owner + 1} hit wall",
                self.config.hit_wall,
            )
        return f"{record.tick:>3}  unknown event", self.config.muted


def show_replay(
    replay: ReplayInfo,
    physics: Physics,
    config: ReplayConfig | None = None,
) -> None:
    ReplayViewer(replay, physics, config).run()


__all__ = ["ReplayConfig", "ReplayViewer", "show_replay"]
