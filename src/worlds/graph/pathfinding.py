from heapq import heappush, heappop
import numpy as np

from utils import distance_euler

distance = distance_euler
def astar(start:"Waypoint", destination:"Waypoint"):
    """A* algorithm."""
    # In the beginning, the start is the only element in our front.
    # NOW, the first element is the total cost through the point, which is
    # the cost from start to point plus the estimated cost to the goal.
    # The second element is the cost of the path from the start to the point.
    # The third element is the position (cell) of the point.
    # The fourth component is the position we came from when entering the tuple to the front.
    from worlds.graph.trafficlights import Waypoint
    assert isinstance(start, Waypoint)
    assert isinstance(destination, Waypoint)
    front = [(distance(np.asarray(start.position), destination.position) + 0.001, 0.001, start, None)]

    # Also, we use a dictionary to remember where we came from.
    visited = {}
    pos = start

    # While there are elements to investigate in our front.
    while front:
        # Get smallest item and remove from front.
        total_cost, cost, pos, previous = heappop(front)

        if pos in visited:
            continue

        # Also remember that we came from previous when we marked pos.
        visited[pos] = previous

        # Check if the goal has been reached.
        if pos == destination:
            break


        # Check all neighbors.
        for street in pos.outgoing:
            # Add to front if: not visited before and no obstacle.
            new_pos = street.destination

            if new_pos not in visited:
                total_cost = cost + street.length()
                dist = distance(np.asarray(new_pos.position), destination.position)
                heappush(front, (total_cost + dist, total_cost, new_pos, pos))

    # Reconstruct path, starting from goal.
    path = []
    if pos == destination:  # If we reached the goal, unwind backwards.
        while pos is not None:
            path.append(pos)
            pos = visited[pos]
        path.reverse()  # Reverse so that path is from start to goal.

    return path