from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Det:
    track_id: int
    class_id: int
    class_name: str
    conf: float
    bbox: list
    centroid: tuple
    area: float
    mask: Optional[list] = None  # polygon [[x, y], ...]
    mask_area: float = 0.0


class BaseDetector(ABC):
    @abstractmethod
    def load(self): ...

    @abstractmethod
    def predict(self, frame) -> List[Det]: ...


class BaseAnalyzer(ABC):
    @abstractmethod
    def update(self, frame_id: int, dets: List[Det]) -> list: ...

    @abstractmethod
    def summary(self) -> list: ...
