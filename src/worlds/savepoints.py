from worlds.graph.trafficlights import Waypoint, SpawnPoint, TrafficLight
from worlds.pixel.world import PixelWorld
from worlds.graph.world import GraphWorld


def pixel_cross() -> PixelWorld:
    world = PixelWorld((100, 100))
    world.add_street((0, 70), (100, 70))  # Horizontal
    world.add_street((50, 0), (50, 100))  # Vertical
    world.add_street((0, 70), (50, 100))  # Diagonal
    world._set_incoming_outgoing_streets()
    return world


def graph_3x3bidirectional() -> GraphWorld:
    world = GraphWorld()

    # 1 2 3
    # 4 5 6
    # 7 8 9

    waypoint1 = TrafficLight((0, 0))
    waypoint2 = TrafficLight((100, 0))
    waypoint3 = TrafficLight((300, 0))

    waypoint4 = TrafficLight((0, 150))
    waypoint5 = TrafficLight((150, 150))
    waypoint6 = TrafficLight((300, 100))

    waypoint7 = TrafficLight((0, 250))
    waypoint8 = TrafficLight((100, 250))
    waypoint9 = TrafficLight((350, 250))

    # Horizontal connections
    waypoint1.connect_to(waypoint2)
    waypoint2.connect_to(waypoint3)

    waypoint4.connect_to(waypoint5)
    waypoint5.connect_to(waypoint6)

    waypoint7.connect_to(waypoint8)
    waypoint8.connect_to(waypoint9)

    # Vertical
    waypoint1.connect_to(waypoint4)
    waypoint2.connect_to(waypoint5)
    waypoint3.connect_to(waypoint6)
    waypoint4.connect_to(waypoint7)
    waypoint5.connect_to(waypoint8)
    waypoint6.connect_to(waypoint9)

    # Diagonal
    waypoint1.connect_to(waypoint5, both_directions=False)
    waypoint5.connect_to(waypoint9, both_directions=False)

    world.add_waypoints(waypoint1, waypoint2, waypoint3, waypoint4, waypoint5, waypoint6, waypoint7, waypoint8,
                        waypoint9)

    # Spawnpoints
    shift = 20
    spawn1 = SpawnPoint(waypoint1.position_relative((-shift, -shift)))
    spawn2 = SpawnPoint(waypoint2.position_relative((0, -shift)))
    spawn3 = SpawnPoint(waypoint3.position_relative((shift*4.5, -shift)))
    spawn4 = SpawnPoint(waypoint6.position_relative((shift*4.5, 0)))
    spawn5 = SpawnPoint(waypoint9.position_relative((shift, shift)))
    spawn6 = SpawnPoint(waypoint8.position_relative((0, shift)))
    spawn7 = SpawnPoint(waypoint7.position_relative((-shift, shift)))
    spawn8 = SpawnPoint(waypoint4.position_relative((-shift, 0)))

    spawn1.connect_to(waypoint1)
    spawn2.connect_to(waypoint2)
    spawn3.connect_to(waypoint3)
    spawn4.connect_to(waypoint6)
    spawn5.connect_to(waypoint9)
    spawn6.connect_to(waypoint8)
    spawn7.connect_to(waypoint7)
    spawn8.connect_to(waypoint4)


    world.add_waypoints(spawn1,spawn2,spawn3,spawn4,spawn5,spawn6,spawn7,spawn8)

    world.validate()

    return world
def graph_3x3circle() -> GraphWorld:
    world = GraphWorld()

    # 1 2 3
    # 4 5 6
    # 7 8 9

    waypoint1 = TrafficLight((0, 0))
    waypoint2 = TrafficLight((100, 0))
    waypoint3 = TrafficLight((300, 0))

    waypoint4 = TrafficLight((0, 150))
    waypoint5 = TrafficLight((150, 150))
    waypoint6 = TrafficLight((300, 100))

    waypoint7 = TrafficLight((0, 250))
    waypoint8 = TrafficLight((100, 250))
    waypoint9 = TrafficLight((350, 250))

    #  Circle connections
    waypoint1.connect_to(waypoint4, both_directions=False)
    waypoint4.connect_to(waypoint7, both_directions=False)
    waypoint7.connect_to(waypoint8, both_directions=False)
    waypoint8.connect_to(waypoint9, both_directions=False)
    waypoint9.connect_to(waypoint6, both_directions=False)
    waypoint6.connect_to(waypoint3, both_directions=False)
    waypoint3.connect_to(waypoint2, both_directions=False)
    waypoint2.connect_to(waypoint1, both_directions=False)

    # Additional connections
    waypoint1.connect_to(waypoint5, both_directions=False)
    waypoint2.connect_to(waypoint5, both_directions=False)
    waypoint3.connect_to(waypoint5, both_directions=False)

    waypoint5.connect_to(waypoint6, both_directions=False)
    waypoint5.connect_to(waypoint8, both_directions=False)
    waypoint5.connect_to(waypoint9, both_directions=False)

    world.add_waypoints(waypoint1, waypoint2, waypoint3, waypoint4, waypoint5, waypoint6, waypoint7, waypoint8,
                        waypoint9)

    # Spawnpoints
    shift = 20
    spawn1 = SpawnPoint(waypoint1.position_relative((-shift, -shift)))
    spawn2 = SpawnPoint(waypoint2.position_relative((0, -shift)))
    spawn3 = SpawnPoint(waypoint3.position_relative((shift*4.5, -shift)))
    spawn4 = SpawnPoint(waypoint6.position_relative((shift*4.5, 0)))
    spawn5 = SpawnPoint(waypoint9.position_relative((shift, shift)))
    spawn6 = SpawnPoint(waypoint8.position_relative((0, shift)))
    spawn7 = SpawnPoint(waypoint7.position_relative((-shift, shift)))
    spawn8 = SpawnPoint(waypoint4.position_relative((-shift, 0)))

    spawn1.connect_to(waypoint1)
    spawn2.connect_to(waypoint2)
    spawn3.connect_to(waypoint3)
    spawn4.connect_to(waypoint6)
    spawn5.connect_to(waypoint9)
    spawn6.connect_to(waypoint8)
    spawn7.connect_to(waypoint7)
    spawn8.connect_to(waypoint4)


    world.add_waypoints(spawn1,spawn2,spawn3,spawn4,spawn5,spawn6,spawn7,spawn8)

    world.validate()

    return world