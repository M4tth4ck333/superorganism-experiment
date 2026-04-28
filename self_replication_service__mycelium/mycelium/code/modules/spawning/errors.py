class SpawnError(Exception):
    """Raised by any spawn pipeline step. `step` identifies which stage failed."""

    def __init__(self, step: str, message: str):
        self.step = step
        super().__init__(f"[{step}] {message}")
