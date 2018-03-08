import numpy as np


def optimize(f, x, learn_rate, n_iter, delta = 0.0001):
    for iter in range(n_iter):
        __grad_step(f, x, learn_rate, delta)
    return x

#Steps inputs x(vector) to function f toward minimums or maximums depending on the value of learn_rate
#(negative for minimization, positive for maximization).
def __grad_step(f, x, learn_rate, delta):
    grad = np.zeros(x.shape, dtype = np.float64)
    f_x = f(x)
    for i in range(0, len(grad)):
        x_add = np.zeros(x.shape)
        x_add[i] = delta
        grad[i] += 0.5*(f(x + x_add) - f_x)/delta
        grad[i] += 0.5*(f_x - f(x - x_add))/delta
    x += learn_rate * grad
    return None
