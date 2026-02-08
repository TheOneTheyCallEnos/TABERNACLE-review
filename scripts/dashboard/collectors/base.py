from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseCollector(ABC):
    """Abstract base class for all collectors."""

    name: str = "base"
    layer_id: int = -1

    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """Collect raw metrics for a layer."""
        raise NotImplementedError
