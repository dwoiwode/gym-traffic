from typing import List

import cv2
import numpy as np

from worlds.pixel import pathfinding
from worlds.pixel.vehicles import PixelVehicle, PixelAudiA6
from worlds.world import World


class PixelWorld(World):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.world = np.zeros(size, dtype=np.float)
        self.vehicles: List[PixelVehicle] = []
        self.streets_incoming = np.asarray([])
        self.streets_outgoing = np.asarray([])

    def add_street(self, pos1, pos2, d_step=0.1):
        pos1 = np.asarray(pos1, dtype=np.float)
        pos2 = np.asarray(pos2, dtype=np.float)
        delta_pos = pos2 - pos1
        street_length = np.linalg.norm(delta_pos)
        steps = street_length / d_step
        delta_pos /= steps
        for i in np.arange(steps):
            pos = np.asarray(pos1 + delta_pos * i, dtype=np.int)
            self.world[pos[0], pos[1]] += 1

    def _set_incoming_outgoing_streets(self):
        street_positions = np.asarray(np.where(self.world > 0)).T
        streets_incoming = street_positions[(street_positions[:, 0] == 0) | (street_positions[:, 1] == 0)]
        streets_outgoing = street_positions[
            (street_positions[:, 0] == self.size[0] - 1) | (street_positions[:, 1] == self.size[1] - 1)]

        self.streets_incoming = streets_incoming
        self.streets_outgoing = streets_outgoing

    def spawn_car(self):
        start = self.streets_incoming[np.random.randint(len(self.streets_incoming))]
        dest = self.streets_outgoing[np.random.randint(len(self.streets_incoming))]

        path = pathfinding.astar(start, dest, self.world)
        if not path:
            return
        # print(path)
        self.vehicles.append(PixelAudiA6(path))

    def step(self, dt=1):
        super().step(dt)
        i = 0
        while i < len(self.vehicles):
            vehicle = self.vehicles[i]
            if len(vehicle.path) == 0:
                self.vehicles.pop(i)
            else:
                vehicle.position = vehicle.path.pop(0)
                i += 1

    def render(self):
        board = np.zeros((self.world.shape[0], self.world.shape[1], 3))
        k = 0.5
        board[:, :, 0] = self.world * k
        board[:, :, 1] = self.world * k
        board[:, :, 2] = self.world * k
        for car in self.vehicles:
            board[car.position] = [0, 0, 1]

        cv2.namedWindow("World", cv2.WINDOW_NORMAL)
        cv2.imshow("World", board.transpose((1, 0, 2)) * 1.)
        cv2.resizeWindow("World", 700, 700)
        if cv2.waitKey(10) == ord("q"):
            exit()