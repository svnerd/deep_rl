import random
import numpy as np


RANDOM_SEED = 100
BATCH_SIZE=128
MAX_STEPS = 200
DISCOUNT_RATE=1

def seed_it():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
