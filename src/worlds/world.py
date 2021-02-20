class World:
    def __init__(self):
        self.t = 0

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def reset(self):
        self.t = 0

    def step(self, dt=1.):
        self.t += dt

    def render(self):
        return str(self)
