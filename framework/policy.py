

class Policy:
    def __init__(self, network):
        self.network = network

    def decide(self):
        return self.network.approximate()

    def update(self, observations):
        self.network.correct(observations)

    def is_good(self):
        return False