import numpy as np

def GINI(sets):
    gini = 0
    for i in range(0, len(sets)):
        proportions = np.bincount(sets[i])
        proportions = (proportions.astype(np.float64) / float(sets[i].shape[0]))
        gini += 1 - np.sum(proportions **2)
    return gini/float(len(sets))
