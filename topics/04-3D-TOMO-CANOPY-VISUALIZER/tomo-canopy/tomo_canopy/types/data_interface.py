from dataclasses import dataclass
from typing import Generic, TypeVar, List

T = TypeVar("T")


@dataclass
class DataLayer(Generic[T]):
    data_points: T
    timestamp: str


V = TypeVar("V")


@dataclass
class DataPoint3D(Generic[V]):
    x: V
    y: V
    z: V


S = TypeVar("S")


@dataclass
class DataLayer3D(DataLayer[list[DataPoint3D[S]]]):
    "Data Layer of 3D data"


@dataclass
class FloatDataLayer3D(DataLayer3D[float]):
    "Float Data Layer of 3D data"


@dataclass
class DataCube:
    als: FloatDataLayer3D
    tls: FloatDataLayer3D
    p_band: FloatDataLayer3D
    gedi: FloatDataLayer3D
    x_dim: int
    y_dim: int
