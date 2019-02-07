import torch

class Agent:
    def __init__(self):
        pass

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename)
        self.network.load_state_dict(state_dict)