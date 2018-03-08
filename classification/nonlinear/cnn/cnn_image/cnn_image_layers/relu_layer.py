from classification.nonlinear.cnn.cnn_image.cnn_image_layers.cnn_layer import CNNLayer
import toolbox.imageop as imageop
import numpy as np

class ReluLayer(CNNLayer):

    def __init__(self, prev_layer = None, next_layer = None):
        CNNLayer.__init__(self, prev_layer, next_layer)

    def _input_grad(self, X, layer_out):
        grad = np.zeros(layer_out.shape + X.shape, dtype = X.dtype)
        greater_than_0_indices = np.argwhere(layer_out > 0).T
        grad[greater_than_0_indices[0], greater_than_0_indices[1], greater_than_0_indices[2], \
        greater_than_0_indices[0], greater_than_0_indices[1], greater_than_0_indices[2]] = 1
        return grad

    def _update_param_grads(self, X, layer_out):
        return None#layer has no parameters

    def step_params(self, learn_rate):
        return None#layer has no parameters

    def _transform(self, X):
        out = X.copy()
        out[out<0]=0
        return out
