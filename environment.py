
class Environment:
    def __init__(self, obstacles, bounds):
        """
        obstacles is a list of Nx2 numpy arrays
        """
        self.obstacles = obstacles
        self.bounds = bounds
