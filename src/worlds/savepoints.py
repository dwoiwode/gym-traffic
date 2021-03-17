from worlds.graph.trafficlights import SpawnPoint, TrafficLight
from worlds.graph.world import GraphWorld


def graph_3x3bidirectional() -> GraphWorld:
    """ 9 Waypoints + 8 Spawnpoints with streets from and to each other. Most are bidirectional streets """
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
    waypoint1.connect_to(waypoint2, both_directions=True)
    waypoint2.connect_to(waypoint3, both_directions=True)

    waypoint4.connect_to(waypoint5, both_directions=True)
    waypoint5.connect_to(waypoint6, both_directions=True)

    waypoint7.connect_to(waypoint8, both_directions=True)
    waypoint8.connect_to(waypoint9, both_directions=True)

    # Vertical
    waypoint1.connect_to(waypoint4, both_directions=True)
    waypoint2.connect_to(waypoint5, both_directions=True)
    waypoint3.connect_to(waypoint6, both_directions=True)
    waypoint4.connect_to(waypoint7, both_directions=True)
    waypoint5.connect_to(waypoint8, both_directions=True)
    waypoint6.connect_to(waypoint9, both_directions=True)

    # Diagonal
    waypoint1.connect_to(waypoint5)
    waypoint5.connect_to(waypoint9)

    world.add_waypoints(waypoint1, waypoint2, waypoint3, waypoint4, waypoint5, waypoint6, waypoint7, waypoint8,
                        waypoint9)

    # Spawnpoints
    shift = 20
    spawn1 = SpawnPoint(waypoint1.position_relative((-shift, -shift)))
    spawn2 = SpawnPoint(waypoint2.position_relative((0, -shift)))
    spawn3 = SpawnPoint(waypoint3.position_relative((shift * 4.5, -shift)))
    spawn4 = SpawnPoint(waypoint6.position_relative((shift * 4.5, 0)))
    spawn5 = SpawnPoint(waypoint9.position_relative((shift, shift)))
    spawn6 = SpawnPoint(waypoint8.position_relative((0, shift)))
    spawn7 = SpawnPoint(waypoint7.position_relative((-shift, shift)))
    spawn8 = SpawnPoint(waypoint4.position_relative((-shift, 0)))

    spawn1.connect_to(waypoint1, both_directions=True)
    spawn2.connect_to(waypoint2, both_directions=True)
    spawn3.connect_to(waypoint3, both_directions=True)
    spawn4.connect_to(waypoint6, both_directions=True)
    spawn5.connect_to(waypoint9, both_directions=True)
    spawn6.connect_to(waypoint8, both_directions=True)
    spawn7.connect_to(waypoint7, both_directions=True)
    spawn8.connect_to(waypoint4, both_directions=True)

    world.add_waypoints(spawn1, spawn2, spawn3, spawn4, spawn5, spawn6, spawn7, spawn8)

    world.validate()

    return world


def graph_3x3circle() -> GraphWorld:
    """ Same as graph_3x3bidirectional but most streets are one-way in a CCW circle """
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
    waypoint1.connect_to(waypoint4)
    waypoint4.connect_to(waypoint7)
    waypoint7.connect_to(waypoint8)
    waypoint8.connect_to(waypoint9)
    waypoint9.connect_to(waypoint6)
    waypoint6.connect_to(waypoint3)
    waypoint3.connect_to(waypoint2)
    waypoint2.connect_to(waypoint1)

    # Additional connections
    waypoint1.connect_to(waypoint5)
    waypoint2.connect_to(waypoint5)
    waypoint3.connect_to(waypoint5)

    waypoint5.connect_to(waypoint6)
    waypoint5.connect_to(waypoint8)
    waypoint5.connect_to(waypoint9)

    world.add_waypoints(waypoint1, waypoint2, waypoint3, waypoint4, waypoint5, waypoint6, waypoint7, waypoint8,
                        waypoint9)

    # Spawnpoints
    shift = 20
    spawn1 = SpawnPoint(waypoint1.position_relative((-shift, -shift)))
    spawn2 = SpawnPoint(waypoint2.position_relative((0, -shift)))
    spawn3 = SpawnPoint(waypoint3.position_relative((shift * 4.5, -shift)))
    spawn4 = SpawnPoint(waypoint6.position_relative((shift * 4.5, 0)))
    spawn5 = SpawnPoint(waypoint9.position_relative((shift, shift)))
    spawn6 = SpawnPoint(waypoint8.position_relative((0, shift)))
    spawn7 = SpawnPoint(waypoint7.position_relative((-shift, shift)))
    spawn8 = SpawnPoint(waypoint4.position_relative((-shift, 0)))

    spawn1.connect_to(waypoint1, both_directions=True)
    spawn2.connect_to(waypoint2, both_directions=True)
    spawn3.connect_to(waypoint3, both_directions=True)
    spawn4.connect_to(waypoint6, both_directions=True)
    spawn5.connect_to(waypoint9, both_directions=True)
    spawn6.connect_to(waypoint8, both_directions=True)
    spawn7.connect_to(waypoint7, both_directions=True)
    spawn8.connect_to(waypoint4, both_directions=True)

    world.add_waypoints(spawn1, spawn2, spawn3, spawn4, spawn5, spawn6, spawn7, spawn8)

    world.validate()

    return world


def graph_cross() -> GraphWorld:
    """ Simple + shaped world """
    #
    #    N
    #  W M E
    #    S >
    #
    world = GraphWorld()
    m = 150
    waypoint_north = SpawnPoint((0, -m), can_end=False)
    waypoint_south = SpawnPoint((0, m), can_end=False)
    waypoint_west = SpawnPoint((-m, 0), can_end=False)
    waypoint_east = SpawnPoint((m, 0), can_end=False)
    waypoint_exit = SpawnPoint((m, m), can_start=False)

    waypoint_middle = TrafficLight((0, 0))

    waypoint_north.connect_to(waypoint_middle)
    waypoint_south.connect_to(waypoint_middle)
    waypoint_west.connect_to(waypoint_middle)
    waypoint_east.connect_to(waypoint_middle)
    waypoint_middle.connect_to(waypoint_exit)

    world.add_waypoints(waypoint_north, waypoint_middle, waypoint_west, waypoint_exit, waypoint_south, waypoint_east)

    world.validate()

    return world


def graph_narrow_tall() -> GraphWorld:
    """ Used as an example image for presentation """
    world = GraphWorld()

    # 1 2
    # 4 5
    # 7 8

    waypoint1 = TrafficLight((0, 0))
    waypoint2 = TrafficLight((100, 0))

    waypoint4 = TrafficLight((0, 150))
    waypoint5 = TrafficLight((150, 150))

    waypoint7 = TrafficLight((0, 250))
    waypoint8 = TrafficLight((100, 250))

    #  Circle connections
    waypoint1.connect_to(waypoint4)
    waypoint4.connect_to(waypoint7)
    waypoint7.connect_to(waypoint8)
    waypoint2.connect_to(waypoint1)

    # Additional connections
    waypoint1.connect_to(waypoint5)
    waypoint2.connect_to(waypoint5)

    waypoint5.connect_to(waypoint8)

    world.add_waypoints(waypoint1, waypoint2, waypoint4, waypoint5, waypoint7, waypoint8)

    # Spawnpoints
    shift = 20
    spawn1 = SpawnPoint(waypoint1.position_relative((-shift, -shift)))
    spawn2 = SpawnPoint(waypoint2.position_relative((3.5 * shift, -shift)))
    spawn4 = SpawnPoint(waypoint5.position_relative((shift, 0)))
    spawn6 = SpawnPoint(waypoint8.position_relative((3.5 * shift, shift)))
    spawn7 = SpawnPoint(waypoint7.position_relative((-shift, shift)))
    spawn8 = SpawnPoint(waypoint4.position_relative((-shift, 0)))

    spawn1.connect_to(waypoint1, both_directions=True)
    spawn2.connect_to(waypoint2, both_directions=True)
    spawn4.connect_to(waypoint5, both_directions=True)
    spawn6.connect_to(waypoint8, both_directions=True)
    spawn7.connect_to(waypoint7, both_directions=True)
    spawn8.connect_to(waypoint4, both_directions=True)

    world.add_waypoints(spawn1, spawn2, spawn4, spawn6, spawn7, spawn8)

    world.validate()

    return world
