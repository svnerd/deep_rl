class SoftEpsilonDecay:
    def __init__(self, start_epsilon, end_epsilon, decay_coeff):
        self.eps = start_epsilon
        self.end = end_epsilon
        self.decay_coeff = decay_coeff

    def decay(self):
        self.eps *= self.decay_coeff
        return max(self.end, self.eps)
