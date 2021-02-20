from worlds.savepoints import graph_3x3circle

if __name__ == '__main__':
    world = graph_3x3circle()
    next_render = 0
    i = 0
    import time
    last_time = int(time.time())
    while True:
        world.step(0.01)
        i += 1
        next_time = int(time.time())
        if last_time != next_time:
            last_time = next_time
            print(f"Updates per second: {i}")
            i = 0
        if world.t >= next_render:
            next_render += 1/10
            world.render()
