from typing import List, Tuple

from utils import kmh_to_ms, zero_to_hundert_in_ms2


class PixelVehicle:
    def __init__(self, path:List[Tuple[int, int]], k_fuel=1.5, max_speed=kmh_to_ms(120), acceleration=zero_to_hundert_in_ms2(14)):
        self.path: List[Tuple[int, int]] = path
        self.position: Tuple[int, int] = self.path[0]
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.velocity = 0
        self.fuel = len(self.path) * k_fuel
        self.size = 5


class PixelAudiA6(PixelVehicle):
    def __init__(self, path: List[Tuple[int, int]]):
        super().__init__(path, max_speed=kmh_to_ms(224))
