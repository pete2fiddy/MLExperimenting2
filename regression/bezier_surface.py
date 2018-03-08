import numpy as np
from matplotlib import pyplot as plt
import regression.bezier_curve as bezier_curve

#P is a D1xD2xD3x...xF where F is the number of features, and the number of D's is the surface input dimension
def surface(P, t):
    if len(P.shape) == 2:
        return bezier_curve.curve(P, t[0])
    P_recurse = np.zeros((P.shape[0], P.shape[len(P.shape)-1]))
    for i in range(0, P.shape[0]):
        P_recurse[i] = surface(P[i], t[:t.shape[0]-1])
    return bezier_curve.curve(P_recurse, t[t.shape[0]-1])


def cost(P, Y, T):
    cost = 0
    for i in range(T.shape[0]):
        cost += np.linalg.norm(Y[i] - surface(P, T[i]))**2
    return cost
