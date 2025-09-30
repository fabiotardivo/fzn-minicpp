import numpy as np

# Initialization here
print("% [ML] Initialization... OK")

# Method to evaluate a *partial assigment*.
# It takes an array of floats where nan=unassigned.
# It returns a float such that the lower the value, the better.
def eval(pa):
    mask = ~np.isnan(pa)          # True where not NaN
    idx = np.argmax(mask)             # index of first True
    val = pa[idx]
    score = len(pa) * idx + val
    #print( f"{pa} -> {score}")
    return score