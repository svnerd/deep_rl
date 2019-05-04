import random
import numpy as np
RANDOM_SEED = 100
BATCH_SIZE = 16

def seed_it():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
