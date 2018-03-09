import numpy as np
import linalg.matrix.matmath as matmath
import test.matrix_math.PLU_decomposition_test as PLU_decomposition_test
import test.matrix_math.determinant_test as determinant_test
import test.matrix_math.inv_test as inv_test
import regression.bezier_curve as bezier_curve
import regression.bezier_surface as bezier_surface
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from mpl_toolkits.mplot3d import Axes3D
from math import pi, sin, cos, sqrt, exp, sqrt
import optimization.gradient_optimize as gradient_optimize

import function.impurity as impurity
from classification.tree.PCA_decision_tree import PCATree
from classification.tree.decision_tree import DecisionTree

import classification.classify_visualize as classify_visualize
from matplotlib import pyplot as plt

from regression.linear_regression import LinearRegression

from classification.linear.linear_regression_classifier import LinearRegressionClassifier

import unsupervised.clustering.connected_spectral_components_clustering as connected_spectral_components_clustering
import function.similarity as similarity

import cv2
import toolbox.imageop as imageop
from classification.nonlinear.cnn.cnn_image.cnn_image_layers.convolution_layer import ConvolutionLayer
from classification.nonlinear.cnn.cnn_image.cnn_image_layers.relu_layer import ReluLayer
from classification.nonlinear.cnn.cnn_image.cnn_image_layers.sigmoid_layer import SigmoidLayer
from classification.nonlinear.cnn.cnn_image.cnn_image_layers.tanh_layer import TanhLayer
from classification.nonlinear.cnn.cnn_image.cnn_image_layers.sum_square_error_layer import SumSquareErrorLayer
from classification.nonlinear.cnn.cnn_image.cnn_image_layers.fully_connected_layer import FullyConnectedLayer
from classification.nonlinear.cnn.cnn_image.cnn_image_layers.softmax_layer import SoftmaxLayer
from classification.nonlinear.cnn.cnn_image.image_cnn import ImageCNN
from sklearn.datasets import load_digits
from mnist import MNIST

'''check to make sure convolution in imageop is correct'''
'''
img = cv2.imread("C:/Users/Peter/Desktop/Free Time CS Projects/Computer Vision Experimenting/images/300 crop 6480x4320.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = img[:12,:12]
img = img.astype(np.float64)

img_volume = np.zeros((img.shape[0], img.shape[1], 1), dtype = np.float64)
img_volume[:,:,0] = img
'''

#CNN Improvements:
#GO THROUGH NET AND MAKE SURE ALL NUMPY SPEEDUPS YIELD THE CORRECT SIZE, ESPECIALLY WHEN += THEM TO A GRADIENT ARRAY
#GRADIENTS FOR LAST PARAMETER LAYER (EITHER ONLY FC OR BOTH CONV AND FC) APPEAR NOT TO MOVE AT ALL
#POSSIBLE NUMERICAL STABILITY ISSUES
#add a nicer total differntial calculator that works for paramter gradients too. (FC layer and conv layer all seem to apply the same
#/similar process)
#completely optimize the methods of CNNLayer first, as every layer uses them and will give the most speed improvements
#add a bias per kernel? May be redundant/bad
#add abstract class for CNN error layers (placed at end)
#make it so that fully connected layers don't need to know their input dimensions by user input. Would be nice if they could
#figure it out themselves
#add an assertion for FC layers that makes sure dimensions are correctly satisfied
mndata = MNIST("C:/Users/Peter/Desktop/Free Time CS Projects/ML Experimenting 2/data/MNIST")
mndata.gz = True
X_old, y_old = mndata.load_training()

X = []
y = []
for i in range(0, len(X_old)):
    #print("X_old[i]: ", X_old[i])
    #if y_old[i] == 0 or y_old[i] == 1:
    img_flat = (np.asarray(X_old[i]).astype(np.float64))/(255.0)
    #print("img_flat: ", img_flat)
    img_square = img_flat.reshape(int(np.sqrt(img_flat.shape[0])), int(np.sqrt(img_flat.shape[0])))
    img_volume = np.zeros(img_square.shape + (1,), dtype = np.float64)
    img_volume[:,:,0] = img_square
    X.append(img_volume)
    iter_y = np.zeros((10,1,1), dtype = np.float64)
    iter_y[y_old[i],0,0] = 1.0
    y.append(iter_y)
    if len(X) >= 100:
        break
    #y.append(img_volume)

X = np.asarray(X)
y = np.asarray(y)

'''
l1 = ConvolutionLayer((5,5), 10, (1,1), (2,2))
l2 = SigmoidLayer()
l3 = ConvolutionLayer((5,5), 10, (2,2), (2,2))
l4 = SigmoidLayer()
l5 = ConvolutionLayer((3,3), 10, (2,2), (1,1))
l6 = SigmoidLayer()
l7 = FullyConnectedLayer(10, (7,7,10))
l8 = SigmoidLayer()
l9 = FullyConnectedLayer(2, (10,1,1))
l10 = SoftmaxLayer()
l11 = SumSquareErrorLayer()

cnn = ImageCNN([l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11])
'''


#maybe problem is need to do chain rule for gradients/weights using THEIR layer's total gradient

l1 = ConvolutionLayer((5,5), 5, (1,1), (2,2))
l2 = ReluLayer()
l3 = ConvolutionLayer((5,5), 1, (2,2), (2,2))
l4 = ReluLayer()
l5 = FullyConnectedLayer(10, (14,14,1))
l6 = ReluLayer()
l7 = FullyConnectedLayer(2, (100,1,1))
l8 = SoftmaxLayer()
l9 = SumSquareErrorLayer()

cnn = ImageCNN([l1,l2,l3,l4,l5,l8,l9])


#working layers: Sigmoid, Softmax, FullyConnectedLayer, CNNLayer, ReluLayer
'''
l1 = FullyConnectedLayer(250, (28,28,1))
l2 = SigmoidLayer()
l3 = FullyConnectedLayer(100, (250,1,1))
l4 = SigmoidLayer()
l5 = FullyConnectedLayer(2, (100,1,1))
l6 = SoftmaxLayer()
l7 = SumSquareErrorLayer()
cnn = ImageCNN([l1,l2,l3,l4,l5,l6,l7])
'''

#cnn_out = cnn.predict(img_volume)
#print("cnn_out; ", cnn_out)





cnn.fit(X, y, learn_rate = -.005, num_steps = 10)


num_right = 0
for i in range(X.shape[0]):
    print("X[i] shape: ", X[i].shape)
    if i%int(X.shape[0]/5) == 0:
        cv2.imshow("X[i]", np.uint8(255*X[i,:,:,0]))
        cv2.waitKey(0)
    prediction = cnn.predict(X[i])
    print("prediction: ", prediction)
    print("np.argmax(prediction): ", np.argmax(prediction))
    print("np.argmax(y[i]): ", np.argmax(y[i]))
    if np.argmax(prediction) == np.argmax(y[i]):
        num_right += 1
print("Total accuracy: " + str(100*num_right/X.shape[0]) + "%")


'''
conv_out = conv_layer.transform(img_volume)
print("conv out shape: ", conv_out.shape)

conv_out_grad = conv_layer.gradient(img_volume, conv_out)
print("conv out grad: ", conv_out_grad.shape)

conv_out_show = np.uint8(255*(conv_out[:,:,0] - conv_out[:,:,0].min())/(conv_out[:,:,0].max() - conv_out[:,:,0].min()))
cv2.imshow("conv_out: ", conv_out_show)
cv2.waitKey(0)
'''



'''
mat = np.array([[1,3,9,2,-1], [1,0,3,-4,3], [0,1,2,3,-1], [-2,3,0,5,4]], dtype = np.float64)
matmath.rref(mat, print_steps = True, stable_swap_rows = False)
'''

'''
def fibbonaci(n):
    v1 = np.array([.5 * (1+np.sqrt(5)), 1])
    v2 = np.array([.5*(1-np.sqrt(5)), 1])
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    P = np.array([v1, v2])
    eigs = np.array([0.5*(1+np.sqrt(5)), 0.5*(1-np.sqrt(5))])**n
    D = np.diag(eigs)
    A = P.dot(D).dot(P.T)
    vec = np.array([1,0])
    return (P.dot(D).dot(P.T).dot(vec))[0]

N = 545
print("fibbonaci " + str(N) + ": " + str(fibbonaci(N)))
'''
'''
perform distance transform on hand mask, then find corners of the distance transform (will be joints) by finding corners using harris corners.

Construct a graph from the vector between joints
'''

'''learn about spectral graph theory'''


'''sparsify distance matrix by checking all pairs to see if there is another pair where projecting onto the subtracted vector between
is less than it's magnitude, the path from point i to j could be broken into a path from point i to point k to point j'''

'''
X, y = datasets.make_moons(n_samples = 500)#datasets.load_iris(return_X_y = True)#
X = X[:,(0,1)]
clusters = connected_spectral_components_clustering.cluster(X, similarity.euclidian, .35)

print("clusters: ", clusters)

for i in range(0, len(clusters)):
    plt.scatter(X[clusters[i], 0], X[clusters[i], 1], c = np.random.rand(3,))
'''
'''
plt.scatter(X[clusters[0], 0], X[clusters[0], 1], color = "red")
plt.scatter(X[clusters[1], 0], X[clusters[1], 1], color = "blue")
'''
plt.show()


'''
X = np.ones((50, 4), dtype = np.float64)
X[:,1] = np.arange(-25, 25, 1, dtype = np.float64)
X[:,2] = X[:,1]**2
X[:,3] = X[:,1]**3
poly_coeffs1 = 3*(np.random.rand(X.shape[1])-.5)
poly_coeffs2 = 3*(np.random.rand(X.shape[1])-.5)
y = np.dot(X, poly_coeffs1)
y += np.random.normal(scale = (y.max()-y.min())/12.0, size = y.shape)
#y = np.array([np.dot(X, poly_coeffs1), np.dot(X,poly_coeffs2)]).T
print("X: ", X)
print("y: ", y)
print("y shape: ", y.shape)

regressor = LinearRegression()
regressor.fit(X, y)

y_hat = regressor.predict(X)

plt.scatter(X[:,1], y, color = "blue")
plt.plot(X[:,1], regressor.predict(X), color = "red")
plt.show()

'''

#TODO: Code k nearest neighbors, KMeans, Linear Descriminant Analysis, Quadratic Descriminant Analysis, Gaussian Descriminant Analysis
#linear loess, neural networks, SVMs,

'''

model = PCATree(1)#LinearRegressionClassifier() #PCATree(1)#DecisionTree()#PCATree(1)
X,y = datasets.load_iris(return_X_y=True)
y += 1
X = X[:, (0,2)]
print("X: ", X)
print("y: ", y)

#model.fit(X, y, linear_model = LinearRegression())
#model.fit(X, y, impurity_func = impurity.GINI, depth = 100, max_split_impurity = .9, min_points_to_split = 5)
model.fit(X, y, impurity_func = impurity.GINI, depth = 10, max_split_impurity = .9, min_points_to_split = 5)

y_hat = model.predict(X)

num_correct = np.count_nonzero(y_hat - y == 0)
print("accuracy: ", float(num_correct)/float(y.shape[0]))


classify_visualize.plot_decision_bounds(X, y, model, mesh_density = 5000)
classify_visualize.plot_data(X[:,(0,1)], y)
plt.show()


#TODO: make general gradient descent take ND-arrays as X instead, as well as allow for extra parameters
#TODO: General-purpose Newton's Method
#TODO: Linear Descriminant Analysis, Quadratic Discriminant Analysis

'''


'''
def f(x):
    return (1-(x[0]**2+x[1]**3))*exp(-(x[0]**2+x[1]**2)/2.0)

X = np.array([np.arange(-3.0, 3.0, .1), np.arange(-3.0, 3.0, .1)]).T
plot_x, plot_y = np.meshgrid(X[:,0], X[:,1])
plot_z = np.zeros(plot_x.shape)
for i in range(0, plot_z.shape[0]):
    for j in range(0, plot_z.shape[1]):
        plot_z[i,j] = f(np.array([plot_x[i,j], plot_y[i,j]]))


fig = plt.figure()
ax = fig.gca(projection = '3d')
surf = ax.plot_wireframe(plot_x, plot_y, plot_z, cmap = cm.RdBu)

x_optimum = np.random.rand(2)*6 - 3
gradient_optimize.optimize(f, x_optimum, 1, 10000, delta = .000001)
print("x optimum: ", x_optimum)
ax.scatter([x_optimum[0]], [x_optimum[1]], [f(x_optimum)], color = "green")

plt.show()
'''

'''
def train_P():
    p = np.array([[1,.5], [0,1], [2,2]])
    N_POINTS = 10
    p = np.random.rand(N_POINTS,2)*5
    T = []
    Y = []

    t = 0

    while t <= 1.01:
        f_t = np.array([t, cos(2.0*pi*t)])
        #f_t = bezier_curve.curve(p, t)
        T.append(t)
        Y.append(f_t)
        t += 0.001
    Y = np.asarray(Y)
    T = np.asarray(T)

    p = bezier_curve.fit(Y, T, n_points = N_POINTS)

    plt.plot(Y[:,0], Y[:,1], color = "blue")
    bezier_curve.graph(p, "red")
    plt.show()
    p_new = np.zeros((p.shape[0], p.shape[1]+1))
    p_new[:,0] = p[:,0]
    p_new[:,2] = p[:,1]
    p = p_new

    P = np.zeros((3, N_POINTS, 3))
    P[0] = p
    P[1] = p + np.array([0,1,1])
    P[2] = p + np.array([0,0,2])

    return P


#TODO:
#General-case gradient descent (uses small changes in input). Attempt to fit bezier surface this way.
# 1D bezier curves for 1d regression
P = train_P()
x = 0
Y = []
T = []
while x <= 1.01:
    y = 0
    while y <= 1.01:
        Y.append(bezier_surface.surface(P, np.array([x,y])))
        y += 0.03
        T.append(np.array([x,y]))
    x += 0.03
Y = np.asarray(Y)
print("Y: ", Y)
print("Y shape: ", Y.shape)
print("y[:, 0] shape: ", Y[:,0].shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Y[:,0], Y[:,1], zs = Y[:,2])

P = gradient_optimize.optimize(bezier_surface.cost, np.random.rand(P.shape[0], P.shape[1], P.shape[2]), -.1, 10000, Y = Y, T = np.asarray(T))



plt.show()
'''
