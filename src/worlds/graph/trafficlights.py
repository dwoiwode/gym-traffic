from typing import List, Tuple, Dict, Optional
import numpy as np

from utils import kmh_to_ms, distance_euler
import warnings


class ValidationError(BaseException):
    pass


class Driveable:
    def __init__(self):
        from worlds.graph.vehicles import Vehicle
        self.vehicles: Dict[Vehicle, float] = {}

    def length(self):
        raise NotImplementedError(f"Requires .length() function for class {self.__class__.__name__}")


    def get_position(self, vehicle:"Vehicle") -> Tuple[float, float]:
        raise NotImplementedError(f"Requires .get_position(vehicle: Vehicle) function for class {self.__class__.__name__}")

    def register_vehicle(self, vehicle: "Vehicle"):
        if vehicle in self.vehicles:
            raise ValueError(f"Vehicle already registered! ({vehicle} on {self})")

        self.vehicles[vehicle] = 0

    def remove_vehicle(self, vehicle: "Vehicle"):
        if vehicle not in self.vehicles:
            raise ValueError(f"{vehicle} does not exists on {self})")

        self.vehicles.pop(vehicle)

    def simulate_move(self, vehicle: "Vehicle", dt=1) -> float:
        return self.vehicles[vehicle] + vehicle.velocity * dt

    def move(self, vehicle: "Vehicle", dt=1) -> bool:
        """ :returns True if finished, else False """
        if vehicle not in self.vehicles:
            raise ValueError(f"{vehicle} does not exists on {self})")

        self.vehicles[vehicle] = self.simulate_move(vehicle, dt)
        if self.vehicles[vehicle] > self.length():
            return True
        return False


class Waypoint(Driveable):
    waypoint_counter = 0

    def __init__(self, position: Tuple[float, float], size: float=20):
        super().__init__()
        Waypoint.waypoint_counter += 1
        self.size = size
        self.id = Waypoint.waypoint_counter
        self.position: Tuple[float, float] = position
        self._outgoing: List[Street] = []
        self._incoming: List[Street] = []

    def __repr__(self):
        return f"{self.__class__.__name__}<{self.id}>({self.position})"

    def move(self, vehicle: "Vehicle", dt=1) -> bool:
        super().move(vehicle, dt)
        return True

    def step(self, dt=1):
        pass

    def length(self):
        return self.size

    def get_position(self, vehicle) -> Tuple[float, float]:
        if vehicle not in self.vehicles:
            raise ValueError(f"{vehicle} does not exists on {self})")
        return self.position

    @property
    def incoming(self) -> List["Street"]:
        return self._incoming

    @property
    def outgoing(self) -> List["Street"]:
        return self._outgoing

    def connect_to(self, destination: "Waypoint", both_directions=True, speed_limit=kmh_to_ms(50)) -> List["Street"]:
        streets = []
        for street in self._outgoing:
            if street.destination == destination:
                break
        else:
            street = Street(self, destination, speed_limit=speed_limit)
            self._outgoing.append(street)
            destination._incoming.append(street)
            streets.append(street)

        if both_directions:
            street = destination.connect_to(self, both_directions=False)
            streets.append(street)

        return streets

    def _validate_self(self) -> bool:
        valid = True
        if len(self.incoming) == 0:
            warnings.warn(f"{self} cannot be reached!")
        elif len(self.outgoing) == 0:
            warnings.warn(f"{self} can be reached but has no way to exit")
            valid = False
        return valid

    def validate(self) -> bool:
        """ Check whether integrity is valid.
        :return True if valid, otherwise False
        """
        # Check self. Use own method for easier inheritance
        valid = self._validate_self()

        # Check outgoing streets
        visited = []
        for street in self.outgoing:
            if not street.validate():
                valid = False

            destination = street.destination
            if street.start != self:
                warnings.warn(f"All outgoing streets has to start at 'self'! Problem at {self} with {street}.")
                valid = False
            if destination in visited:
                warnings.warn(f"There is more than one connection from {self} to {destination}!")
                valid = False
            if street not in destination._incoming:
                warnings.warn(f"{street} is registered at {self} but not at {destination}!")
                valid = False
            visited.append(destination)

        # Check incoming streets
        visited = []
        for street in self.incoming:
            start = street.start
            if street not in start._outgoing:
                warnings.warn(f"{street} is registered at {self} but not at {start}!")
                valid = False
            if start in visited:
                warnings.warn(f"There is more than one connection from {start} to {self}!")
                valid = False
            visited.append(start)

        return valid

    def streets_to(self, waypoint:"Waypoint"):
        return list(filter(lambda s: s.destination == waypoint, self.outgoing))

    def position_relative(self, relative):
        x, y = self.position
        return x + relative[0], y + relative[1]

    def is_allowed_to_drive_from(self, street:"Street"):
        return True


class SpawnPoint(Waypoint):
    def __init__(self, position: Tuple[float, float], can_start=True, can_end=True):
        super().__init__(position, 0.1)
        self.can_start = can_start
        self.can_end = can_end


class TrafficLight(Waypoint):
    def __init__(self, position: Tuple[float, float], size:float=20, transition_duration:float=20):
        super().__init__(position,size=size)
        self.green = []
        self.transition_duration = transition_duration

    def step(self, dt=1):
        super(TrafficLight, self).step()
        # self.green = self.incoming
        if np.random.random() < 0.01:
            self.green = np.random.choice(self.incoming, 1)

    def is_allowed_to_drive_from(self, street:"Street"):
        return self.is_green(street)

    def is_green(self, street:"Street"):
        return street in self.green


class Street(Driveable):
    def __init__(self, start: Waypoint, dest: Waypoint, lanes=1, speed_limit=kmh_to_ms(50)):
        super().__init__()
        self.start: Waypoint = start
        self.destination: Waypoint = dest
        self.lanes: int = lanes
        self.speed_limit: float = speed_limit

    def __repr__(self):
        return f"{self.__class__.__name__}({self.start}, {self.destination}, lanes={self.lanes})"

    def length(self):
        return distance_euler(np.asarray(self.start.position), self.destination.position)

    def get_position(self, vehicle) -> Tuple[float, float]:
        if vehicle not in self.vehicles:
            raise ValueError(f"{vehicle} does not exists on {self})")
        p1 = np.asarray(self.start.position)
        p2 = np.asarray(self.destination.position)
        street = p2 - p1
        street = street / np.linalg.norm(street)
        pos = p1 + self.vehicles[vehicle] * street
        return tuple(pos)

    @property
    def orientation(self):
        x,y = np.asarray(self.destination.position) - self.start.position
        return np.arctan2(x,y)

    def validate(self):
        valid = True
        if self not in self.start.outgoing:
            warnings.warn(f"{self} is not registered in {self.start}!")
            valid = False

        if self not in self.destination.incoming:
            warnings.warn(f"{self} is not registered in {self.destination}!")
            valid = False

        if self.lanes < 1:
            warnings.warn(f"{self} has no lanes! (lanes={self.lanes})")
            valid = False

        return valid

    def opposite_direction(self) -> Optional["Street"]:
        for street in self.destination.outgoing:
            if street.destination == self.start:
                return street
        return None

