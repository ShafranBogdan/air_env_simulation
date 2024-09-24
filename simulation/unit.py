from simulation.time import Time

class Unit:
    def __init__(self) -> None:
        self.time = Time()

    def trigger(self, **kwargs) -> None:
        raise NotImplementedError()