from typing import Iterable, Tuple


class SyncAdapter:
    """Strategy placeholder for multi-node sync of history."""

    def push(self, rows: Iterable[Tuple]) -> None:
        """Push rows to remote nodes."""
        return None

    def pull(self) -> Iterable[Tuple]:
        """Pull rows from remote nodes."""
        return []
