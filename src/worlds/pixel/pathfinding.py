import numpy as np
from heapq import heappush, heappop

from utils import distance_euler


distance = distance_euler


def astar(start, goal, streets):
    """A* algorithm."""
    # In the beginning, the start is the only element in our front.
    # NOW, the first element is the total cost through the point, which is
    # the cost from start to point plus the estimated cost to the goal.
    # The second element is the cost of the path from the start to the point.
    # The third element is the position (cell) of the point.
    # The fourth component is the position we came from when entering the tuple to the front.
    front = [(distance(start, goal) + 0.001, 0.001, start, None)]
    movements = [(1, 0, 1.), (0, 1, 1.), (-1, 0, 1.), (0, -1, 1.)]

    # In the beginning, no cell has been visited.
    extents = streets.shape
    visited = np.zeros(extents, dtype=np.float32)

    # Also, we use a dictionary to remember where we came from.
    came_from = {}

    # While there are elements to investigate in our front.
    # YOUR CODE HERE
    while front:
        # Get smallest item and remove from front.
        element = heappop(front)

        # Check if this has been visited already.
        # CHANGE 01_e: use the following line as shown.
        total_cost, cost, pos, previous = element
        pos = tuple(pos)
        if visited[pos] > 0:
            continue

        # Now it has been visited. Mark with cost.
        visited[pos] = cost

        # Also remember that we came from previous when we marked pos.
        came_from[pos] = previous

        # Check if the goal has been reached.
        if pos == tuple(goal):
            break  # Finished!

        # Check all neighbors.
        for dx, dy, deltacost in movements:
            # Determine new position and check bounds.
            new_x = pos[0] + dx
            new_y = pos[1] + dy
            if not (0 <= new_x < extents[0]):
                continue
            if not (0 <= new_y < extents[1]):
                continue

            # Add to front if: not visited before and no obstacle.
            new_pos = (new_x, new_y)

            if visited[new_pos] == 0 and streets[new_pos] > 0:
                total_cost = cost + deltacost
                dist = distance(new_pos, goal)
                heappush(front, (total_cost + dist, total_cost, new_pos, pos))

    # Reconstruct path, starting from goal.
    path = []
    if pos == tuple(goal):  # If we reached the goal, unwind backwards.
        while pos:
            path.append(pos)
            pos = came_from[pos]
        path.reverse()  # Reverse so that path is from start to goal.

    return path