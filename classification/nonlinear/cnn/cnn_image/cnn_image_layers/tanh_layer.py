from classification.nonlinear.cnn.cnn_image.cnn_image_layers.cnn_layer import CNNLayer
import toolbox.imageop as imageop
import numpy as np

class TanhLayer(CNNLayer):

    def __init__(self, prev_layer = None, next_layer = None):
        CNNLayer.__init__(self, prev_layer, next_layer)

    def _input_grad(self, X, layer_out):
        grad = np.zeros(layer_out.shape+X.shape, dtype = X.dtype)
        wrt_output_grad = 1-layer_out**2
        ijk_indices = np.argwhere(wrt_output_grad >= wrt_output_grad.min()).T#kind of a redundant way of doing it but should be an easy and
        #fast way to get all indices of wrt_output_grad

        grad[ijk_indices[0], ijk_indices[1], ijk_indices[2], ijk_indices[0], ijk_indices[1], ijk_indices[2]] = \
        wrt_output_grad[ijk_indices[0], ijk_indices[1], ijk_indices[2]]
        #np.testing.assert_equal(grad, self._input_grad_old(X, layer_out))
        #print("sigmoid input grad max: ", grad.max())
        return grad


    def _update_param_grads(self, X, layer_out):
        return None#layer has no parameters

    def step_params(self, learn_rate):
        return None#layer has no parameters

    def _transform(self, X):
        #make more numerically stable. Easiest way is to just put in terms of sigmoid's transform, since
        #that one uses log-sum-exp trick for stability
        return (2.0/(1.0+np.exp(-2*X)))-1
