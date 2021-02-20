from typing import Dict, Tuple
from matplotlib import cm


import numpy as np
import cv2


class GraphImageRenderer:
    def __init__(self, world:"GraphWorld", pixel_per_meter=5, waypoint_radius=10, street_width=2):
        self.world = world
        self.pixel_per_meter = pixel_per_meter
        self.waypoint_radius = waypoint_radius
        self.street_width = street_width


        self._min_point_m = 0,0

        self.static_img = np.zeros((10,10))

    def update_static_img(self):
        pass


    def render(self, update_static=False):
        # == Preparing ==
        waypoint_coordinates = []
        streets = []
        street_wp_to_index: Dict[Tuple[int, int], int] = {}
        street_colors = []
        for waypoint in self.world.waypoints:
            waypoint_coordinates.append(waypoint.position)

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
                orthonormal_vector = np.tile(orthogonal_vector / np.linalg.norm(orthogonal_vector), self.street_width)
                first_lane_offset = street.lanes - n_lanes + 1
                # first_lane_offset = street.lanes - n_lanes - 2.3
                color_ind = 1 - min(1., len(street.vehicles) * 5 / len(street))
                for lane in range(street.lanes):
                    translation = orthonormal_vector * (first_lane_offset + lane) * self.street_width
                    streets.append(values + translation)
                    street_colors.append(color_ind)
                street_wp_to_index[street_identifier] = idx

        vehicles = [vehicle.position for vehicle in self.world.vehicles]

        # Determine size
        waypoint_coordinates = np.asarray(waypoint_coordinates, dtype=np.float32)
        min_pos_m = np.min(waypoint_coordinates, axis=0) - 1.5 * self.waypoint_radius
        size = np.max(waypoint_coordinates - min_pos_m, axis=0) + 1.5 * self.waypoint_radius

        # Calculate meter to pixel coordinates
        waypoint_coordinates = np.asarray((waypoint_coordinates - min_pos_m) * self.pixel_per_meter, dtype=np.int)
        streets = np.asarray((np.asarray(streets) - np.tile(min_pos_m, self.street_width)) * self.pixel_per_meter, dtype=np.int)
        if vehicles:
            vehicle_coordinates = np.asarray((np.asarray(vehicles) - min_pos_m) * self.pixel_per_meter, dtype=np.int)
        else:
            vehicle_coordinates = np.zeros((0, self.street_width))
        self.waypoint_radius = int(self.waypoint_radius * self.pixel_per_meter)
        size = np.asarray(size * self.pixel_per_meter, dtype=np.int)
        street_width = int(self.street_width * self.pixel_per_meter)

        # == Drawing ==
        img = np.zeros((size[1], size[0], 3), dtype=np.float32)
        # Streets
        for i, (x1, y1, x2, y2) in enumerate(streets):
            color_value = street_colors[i]
            color = cm.get_cmap("RdYlGn")(color_value)[:3][::-1]
            cv2.line(img, (x1, y1), (x2, y2), color=color, thickness=max(1, street_width))
            cv2.arrowedLine(img, (x1, y1), (x2 - (x2 - x1) // self.street_width, y2 - (y2 - y1) // self.street_width), color=color,
                            thickness=max(street_width // 4, 1))
        # for x1, y1, x2, y2 in streets:
        #     cv2.line(img, (x1, y1), (x2, y2), color=[0,0,0], thickness=max(1, street_width // 20),lineType=cv2.LINE_8)

        # Waypoints
        for i, center in enumerate(waypoint_coordinates):
            cv2.circle(img, tuple(center), self.waypoint_radius, color=[0, 0, 0], thickness=-1)
            color = [1, 1, 1]
            cv2.circle(img, tuple(center), self.waypoint_radius, color=color, thickness=int(self.pixel_per_meter * 0.5))
            # cv2.addText(img, str(i), center, cv2.FONT_HERSHEY_PLAIN, px_per_meter, color)

        # Vehicles
        for vehicle in vehicle_coordinates:
            color = [1, 0, 0]
            cv2.circle(img, tuple(vehicle), max(1, self.waypoint_radius // self.street_width), color, thickness=-1)