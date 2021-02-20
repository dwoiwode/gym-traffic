from typing import List, Tuple
import numpy as np

from worlds.graph.trafficlights import Street, Waypoint, Driveable
from utils import kmh_to_ms, zero_to_hundert_in_ms2


class Vehicle:
    def __init__(self, path: List[Waypoint], k_fuel=1.5, max_speed=kmh_to_ms(120),
                 acceleration=zero_to_hundert_in_ms2(14)):
        # Simulation parameters
        self.path: List[Waypoint] = path
        self.current_street: Driveable = self.path.pop(0)
        self.current_street.register_vehicle(self)

        # Vehicle stats
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.velocity = 0
        self.fuel = 10000 * k_fuel  # TODO: Calculate based on pathlength
        self.length = 5

    @property
    def position(self):
        return self.current_street.get_position(self)

    def can_drive(self, dt=1):
        foreseeing = 2
        own_position = self.current_street.simulate_move(self, 0)
        next_position = self.current_street.simulate_move(self, foreseeing)

        # Check fuel
        if self.fuel <= 0:
            return False

        # Check Traffic lights
        if isinstance(self.current_street, Street):
            crossing = self.current_street.destination
            on_crossing = self.current_street.length() - crossing.length() / 2
            if own_position < on_crossing <= next_position:
                return crossing.is_allowed_to_drive_from(self.current_street)

        # Check other vehicles
        for vehicle in self.current_street.vehicles:
            vehicle_position = self.current_street.simulate_move(vehicle, 0)
            vehicle_next_position = self.current_street.simulate_move(vehicle, foreseeing/2)
            if own_position < vehicle_position:  # Other Vehicle is in front of us
                if next_position + self.length / 2 > vehicle_next_position - vehicle.length/2:
                    return False

        return True

    def step(self, dt=1) -> bool:
        """

        :return: True if vehicle reached destination, False otherwise
        """
        if self.velocity < self.max_speed:
            self.velocity += self.acceleration * dt
        if not self.can_drive(dt):
            if self.velocity > 0:
                self.velocity = max(0, self.velocity - 20 * self.acceleration * dt)
            # return False

        reached_end = self.current_street.move(self, dt)
        if reached_end:
            self.current_street.remove_vehicle(self)
            if len(self.path) == 0:
                return True

            if isinstance(self.current_street, Waypoint):
                # Todo: Choose street based on fullness
                self.current_street = np.random.choice(self.current_street.streets_to(self.path[0]))
            elif isinstance(self.current_street, Street):
                assert self.current_street.destination == self.path[0]
                self.current_street = self.path.pop(0)

            self.current_street.register_vehicle(self)

        return False


class AudiA6(Vehicle):
    def __init__(self, path: List[Waypoint]):
        super().__init__(path, max_speed=kmh_to_ms(224))
