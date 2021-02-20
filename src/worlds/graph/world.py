from typing import List, Dict, Tuple

import cv2
print("OpenCV Version:", cv2.__version__)
import numpy as np
from matplotlib import cm


from worlds.graph.pathfinding import astar
from worlds.graph.renderer import GraphImageRenderer
from worlds.graph.trafficlights import Waypoint, SpawnPoint, TrafficLight
from worlds.graph.vehicles import Vehicle
from worlds.world import World


class GraphWorld(World):
    def __init__(self):
        super().__init__()
        self._waypoints: List[Waypoint] = []
        self._vehicles: List[Vehicle] = []

        self._renderer = GraphImageRenderer(self)

    @property
    def waypoints(self):
        return self._waypoints

    @property
    def start_waypoints(self):
        return list(filter(lambda w: isinstance(w, SpawnPoint) and w.can_start, self.waypoints))

    @property
    def destination_waypoints(self):
        return list(filter(lambda w: isinstance(w, SpawnPoint) and w.can_end, self.waypoints))

    @property
    def vehicles(self):
        return self._vehicles

    def step(self, dt=1.):
        super().step(dt)
        removable_vehicles = []
        for waypoint in self.waypoints:
            waypoint.step(dt)
        for vehicle in self.vehicles:
            if vehicle.step(dt):
                removable_vehicles.append(vehicle)

        for vehicle in removable_vehicles:
            self.remove_vehicle(vehicle)

        min_cars = 0.5
        if np.random.random() < min_cars / (len(self.vehicles) + 0.00001):
            self.spawn_vehicle()

    def add_waypoints(self, *waypoints: Waypoint):
        for waypoint in waypoints:
            if waypoint not in self.waypoints:
                self._waypoints.append(waypoint)

    def spawn_vehicle(self):
        path = []
        while len(path) == 0:
            start = np.random.choice(self.start_waypoints)
            destination = np.random.choice(self.destination_waypoints)

            path = astar(start, destination)

        vehicle = Vehicle(path)
        self.vehicles.append(vehicle)

    def remove_vehicle(self, vehicle:Vehicle):
        self._vehicles.remove(vehicle)

    def validate(self):
        valid = True
        for wp in self.waypoints:
            if not wp.validate():
                valid = False

        return valid

    def render(self, mode='human'):
        # == Constants ==
        px_per_meter = 3
        street_width = 2  # m
        border_offset = [0, 75]  # px
        to_deg = 180 / np.pi

        # == Preparing ==
        waypoint_coordinates:List[Tuple[float, float]] = []
        waypoint_radii:List[float] = []
        waypoint_traffic_lights:List[List[bool, float, float]] = []
        streets:List[Tuple[float, float, float, float]] = []
        street_wp_to_index: Dict[Tuple[int, int], int] = {}
        street_colors:List[float] = []
        for waypoint in self.waypoints:
            waypoint_coordinates.append(waypoint.position)
            waypoint_radii.append(waypoint.length()/2)

            if isinstance(waypoint, TrafficLight):
                traffic_lights = []
                for street in waypoint.incoming:
                    if waypoint.is_green(street):
                        can_drive = True
                    else:
                        can_drive = False

                    orientation = street.orientation
                    traffic_lights.append([can_drive, orientation])
                waypoint_traffic_lights.append(traffic_lights)
            else:
                waypoint_traffic_lights.append([])
            for street in waypoint.outgoing:
                street_identifier = (waypoint.id, street.destination.id)
                if street_identifier in street_wp_to_index:
                    continue
                n_lanes = street.lanes
                opposite_lane = street.opposite_direction()
                if opposite_lane is not None:
                    n_lanes += opposite_lane.lanes

                idx = len(streets)
                x1, y1 = street.start.position
                x2, y2 = street.destination.position
                values = np.asarray([x1, y1, x2, y2])
                if y2 - y1 == 0:
                    orthogonal_vector = np.asarray([0, 1])
                else:
                    orthogonal_vector = np.asarray([1, (x2 - x1) / (y2 - y1)])
                orthonormal_vector = np.tile(orthogonal_vector / np.linalg.norm(orthogonal_vector), 2)
                # orthonormal_vector = np.asarray([0,0,0,0])
                first_lane_offset = street.lanes - n_lanes + 1
                # first_lane_offset = street.lanes - n_lanes - 2.3
                color_ind = 1 - min(1., len(street.vehicles) * 20 / street.length())
                for lane in range(street.lanes):
                    translation = orthonormal_vector * (first_lane_offset + lane) * street_width
                    streets.append(values + translation)
                    street_colors.append(color_ind)
                street_wp_to_index[street_identifier] = idx

        vehicles = [vehicle.position for vehicle in self.vehicles]
        vehicle_sizes = [vehicle.length for vehicle in self.vehicles]

        # == Transform meter into pixel ==
        # Determine size
        waypoint_coordinates = np.asarray(waypoint_coordinates, dtype=np.float32)
        waypoint_radii = np.asarray(waypoint_radii, dtype=np.float32)
        min_pos_m = np.min(waypoint_coordinates - np.reshape(waypoint_radii, (-1,1)), axis=0)
        size = np.max(waypoint_coordinates - min_pos_m + np.reshape(waypoint_radii, (-1,1)), axis=0)

        # Calculate meter to pixel coordinates
        waypoint_coordinates = np.asarray((waypoint_coordinates - min_pos_m) * px_per_meter, dtype=np.int) + border_offset
        streets = np.asarray((np.asarray(streets) - np.tile(min_pos_m, 2)) * px_per_meter, dtype=np.int) + np.tile(border_offset, 2)
        if vehicles:
            vehicle_coordinates = np.asarray((np.asarray(vehicles) - min_pos_m) * px_per_meter, dtype=np.int) + border_offset
            vehicle_sizes = np.asarray(vehicle_sizes * px_per_meter, dtype=np.int)
        else:
            vehicle_coordinates = np.zeros((0,2))
            vehicle_sizes = np.zeros((0,2))
        waypoint_radii = np.asarray(waypoint_radii * px_per_meter, np.int)
        size = np.asarray(size * px_per_meter, dtype=np.int)  + border_offset
        street_width = int(street_width * px_per_meter)

        # == Drawing ==
        img = np.zeros((size[1], size[0], 3), dtype=np.float32)
        img[border_offset[1]] = (1,1,1)
        cv2.rectangle(img, [border_offset, size-border_offset],color=(1,1,1),thickness=10)

        # Streets
        for i, (x1, y1, x2, y2) in enumerate(streets):
            color_value = street_colors[i]
            color = cm.get_cmap("RdYlGn")(color_value)[:3][::-1]
            cv2.line(img, (x1, y1), (x2, y2), color=color, thickness=max(1, street_width))
            cv2.arrowedLine(img, (x1, y1), (x2 - (x2 - x1) // 2, y2 - (y2 - y1) // 2), color=color,
                            thickness=max(street_width // 4, 1))
        # for x1, y1, x2, y2 in streets:
        #     cv2.line(img, (x1, y1), (x2, y2), color=[0,0,0], thickness=max(1, street_width // 20),lineType=cv2.LINE_8)

        # Waypoints
        for i, (center, waypoint_radius, traffic_lights) in enumerate(zip(waypoint_coordinates, waypoint_radii, waypoint_traffic_lights)):
            cv2.circle(img, tuple(center), waypoint_radius, color=[0, 0, 0], thickness=-1)
            color = [1, 1, 1]
            cv2.circle(img, tuple(center), waypoint_radius, color=color, thickness=int(px_per_meter * 0.5))
            for can_drive, orientation in traffic_lights:
                color = [0,1,0] if can_drive else [0,0,1]
                orientation = orientation
                new_center = center - np.asarray([0,waypoint_radius]) @ [[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]]
                cv2.circle(img, (int(new_center[0]), int(new_center[1])), 5, color, thickness=-1)
                # cv2.ellipse(img, center, waypoint_radius, (int(start), int(end)), color=color,thickness=int(px_per_meter * 0.6))
            # cv2.addText(img, str(i), center, cv2.FONT_HERSHEY_PLAIN, px_per_meter, color)

        # Vehicles
        for vehicle, vehicle_size in zip(vehicle_coordinates, vehicle_sizes):
            color = [1, 0, 0]
            cv2.circle(img, tuple(vehicle), max(1, vehicle_size),color,thickness=-1)

        # Status
        n_cars = len(vehicle_sizes)
        t = self.t
        traffic_flow = np.nan
        mean_velocity = np.mean([vehicle.velocity for vehicle in self.vehicles])

        text = f"t={t: 9.3f}s, cars={n_cars:3d}, flow={traffic_flow}, mean_vel={mean_velocity:6.4f} m/s"
        cv2.putText(img, text,(0, 20),cv2.FONT_HERSHEY_PLAIN,2, color=(1,1,1), bottomLeftOrigin=False)

        if mode == 'human':
            cv2.imshow("World", img)
            if cv2.waitKey(10) == ord("q"):
                cv2.destroyWindow("World")
                exit(0)
