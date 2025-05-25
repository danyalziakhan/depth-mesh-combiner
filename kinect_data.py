from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class SensorsConfiguration:
    SENSOR1_TOP: int = 75
    SENSOR1_BOTTOM: int = 292
    SENSOR1_LEFT: int = 30
    SENSOR1_RIGHT: int = 397

    SENSOR2_TOP: int = 94
    SENSOR2_BOTTOM: int = 328
    SENSOR2_LEFT: int = 121
    SENSOR2_RIGHT: int = 489

    SENSOR3_TOP: int = 165
    SENSOR3_BOTTOM: int = 362
    SENSOR3_LEFT: int = 24
    SENSOR3_RIGHT: int = 407

    SENSOR4_TOP: int = 124
    SENSOR4_BOTTOM: int = 326
    SENSOR4_LEFT: int = 73
    SENSOR4_RIGHT: int = 435

    SENSOR1_ANGLE: int = 4
    SENSOR2_ANGLE: int = 0
    SENSOR3_ANGLE: int = 0
    SENSOR4_ANGLE: int = 2

    LEFT_MARGIN: int = 89
    RIGHT_MARGIN: int = 104
    TOP_MARGIN: int = 56
    BOTTOM_MARGIN: int = 39

    MIN_DEPTH_VALUE: int = 3900
    MAX_DEPTH_VALUE: int = 4400

    def get_midpoint(self):
        return (self.MIN_DEPTH_VALUE + self.MAX_DEPTH_VALUE) // 2

    def __getitem__(self, key):
        values = [getattr(self, f.name) for f in fields(self)]
        return values[key]

    def __setitem__(self, key, value):
        names = [f.name for f in fields(self)]
        if isinstance(key, int):
            setattr(self, names[key], value)
        elif isinstance(key, slice):
            for name, val in zip(names[key], value):
                setattr(self, name, val)
