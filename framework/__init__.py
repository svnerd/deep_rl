


# DRL algo so far are just 2 parts:
# 1. starting from an random policy to interact with env and generate samples.
# 2. use the sample to estimate either policy or state value.
# 3. use the esitmation to correct policy or state value to make it better.
# 4. use the updated or fixed policy to generate samples to advance towards better estimation.
# 5. state value can be viewed as policy as well because there are only 2 methods to convert state value to action,
#   namely argmax and espilon-greedy.
# 6. estimation correction is based off deep neaual network.

