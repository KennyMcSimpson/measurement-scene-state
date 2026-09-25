"""Small immutable configuration contracts for the first dynamic carrier."""

from dataclasses import dataclass
from math import isfinite, prod


@dataclass(frozen=True)
class CarrierConfig:
    feature_dim: int = 8
    hidden_dim: int = 8
    expansion_dim: int = 16
    grid_size: tuple[int, int, int] = (8, 8, 8)
    local_bounds_m: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-6.0, -4.0, 0.05),
        (6.0, 4.0, 12.05),
    )
    token_count: int = 128

    def __post_init__(self):
        if any(
            value <= 0
            for value in (self.feature_dim, self.hidden_dim, self.expansion_dim, self.token_count)
        ):
            raise ValueError("Carrier widths and token_count must be positive")
        if len(self.grid_size) != 3 or min(self.grid_size) < 2:
            raise ValueError("grid_size must have three dimensions of at least two")
        if self.token_count > prod(self.grid_size):
            raise ValueError("token_count cannot exceed voxel count")
        if len(self.local_bounds_m) != 2 or any(len(v) != 3 for v in self.local_bounds_m):
            raise ValueError("local_bounds_m must be [2,3]")
        for lower, upper in zip(*self.local_bounds_m, strict=True):
            if not isfinite(lower) or not isfinite(upper) or upper <= lower:
                raise ValueError("Invalid metric bounds")

    @property
    def statistics_dim(self):
        return 2 * self.feature_dim + 7


@dataclass(frozen=True)
class WriteConfig:
    learning_rate: float = 0.01
    max_update_norm: float = 0.05

    def __post_init__(self):
        if not all(isfinite(v) and v > 0 for v in (self.learning_rate, self.max_update_norm)):
            raise ValueError("Write rate and norm limit must be finite and positive")
