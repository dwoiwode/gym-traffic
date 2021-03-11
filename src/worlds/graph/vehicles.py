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

    @property
    def next_street(self):
        if isinstance(self.current_street, Waypoint):
            # Todo: Choose lane based on fullness
            return np.random.choice(self.current_street.streets_to(self.path[0]))
        elif isinstance(self.current_street, Street):
            assert self.current_street.destination == self.path[0]
            return self.path[0]

    def can_drive(self, dt=1):
        foreseeing = 2.
        own_position = self.current_street.simulate_move(self, 0)
        next_position = self.current_street.simulate_move(self, foreseeing) + self.length / 2

        # Check fuel
        if self.fuel <= 0:
            return False

        # Check Traffic lights
        if isinstance(self.current_street, Street):
            next_crossing = self.current_street.destination
            on_next_crossing = self.current_street.length() - next_crossing.length() / 2
            if own_position < on_next_crossing <= next_position:  # Drives onto crossing next step
                if not next_crossing.is_allowed_to_drive_from(self.current_street):  # is_green
                    return False
            elif next_position > self.current_street.length():
                next_street = self.next_street
                return next_street is None or not next_street.will_collide(self, foreseeing=foreseeing)

        # Check other vehicles in street
        return not self.current_street.will_collide(self, foreseeing=foreseeing)

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

            next_street = self.next_street
            if isinstance(next_street, Waypoint):
                self.path.pop(0)
            self.current_street = next_street

            self.current_street.register_vehicle(self)

        return False


class AudiA6(Vehicle):
    def __init__(self, path: List[Waypoint]):
        super().__init__(path, max_speed=kmh_to_ms(224))
