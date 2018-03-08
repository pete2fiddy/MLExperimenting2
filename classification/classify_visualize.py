import numpy as np
from matplotlib import pyplot as plt


def plot_data(X, y):
    assert X.shape[1] == 2
    plt.scatter(X[:,0], X[:,1], c = y, cmap = plt.cm.coolwarm)

def plot_decision_bounds(X, y, model, mesh_density = 500, fill = True, alpha = .5, transform_func = None):
    assert X.shape[1] == 2
    if transform_func is not None:
        X = transform_func(X)
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    mesh_step = ((x_max-x_min) + (y_max-y_min))/mesh_density
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step),
                         np.arange(y_min, y_max, mesh_step))

    X_mesh = (np.c_[xx.ravel(), yy.ravel()])

    Z = model.predict((np.c_[xx.ravel(), yy.ravel()]))

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    if fill:
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha = alpha)
    else:
        plt.contour(xx, yy, Z, cmap = plt.cm.coolwarm, alpha = alpha)
