class AntennaPatch:
    def __init__(self, fo):
        self.frequency = fo
        self.geometry = {}

    def create_dielectric(self):
        raise NotImplementedError

    def create_patch(self):
        raise NotImplementedError

    def create_ground(self):
        raise NotImplementedError

    def create_feed(self):
        raise NotImplementedError

    def get_geometry(self):
        return self.geometry

    def visualize(self):
        raise NotImplementedError
