PLAYER_RADIUS = 0.5
PLAYER_DIAMETER = 2 * PLAYER_RADIUS

class Physics:
    def __init__(
        self,
        board_x: float,
        board_y: float,
        player_speed: float,
        bullet_size: float,
        bullet_speed_ratio: float,
        match_duration: int,
    ):
        if (
            board_x < PLAYER_DIAMETER
            or board_y < PLAYER_DIAMETER
            or player_speed < 0
            or bullet_speed_ratio < 0
            or bullet_size < 0
            or max(board_x, board_y) < 2 * PLAYER_DIAMETER
            or match_duration <= 0
        ):
            raise ValueError("Invalid args")

        self.board_x = board_x
        self.board_y = board_y
        self.player_speed = player_speed
        self.bullet_size = bullet_size
        self.bullet_speed = bullet_speed_ratio * player_speed
        self.match_duration = match_duration