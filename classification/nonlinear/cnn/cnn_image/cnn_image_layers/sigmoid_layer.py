from classification.nonlinear.cnn.cnn_image.cnn_image_layers.cnn_layer import CNNLayer
import toolbox.imageop as imageop
import numpy as np

class SigmoidLayer(CNNLayer):

    def __init__(self, prev_layer = None, next_layer = None):
        CNNLayer.__init__(self, prev_layer, next_layer)

    #returns a 6d arr grad, where grad[i,j,k,w,h,d] = partial(this layer[i,j,k])/partial(input[w,h,d])
    #uses the input to this layer, or layer out if necessary, to calculate.
    def _input_grad(self, X, layer_out):
        grad = np.zeros(layer_out.shape+X.shape, dtype = X.dtype)
        wrt_output_grad = layer_out*(1.0-layer_out)
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
        #uses log-exp-sum trick to calculate sigmoid
        out = np.zeros(X.shape, dtype = np.float64)
        pos_expX = np.exp(X)
        neg_expX = np.exp(-X)
        add_to_out1 = 1.0/(1.0 + neg_expX)
        add_to_out2 = pos_expX/(1+pos_expX)
        add_to_out1[X < 0] = 0
        add_to_out2[X >= 0] = 0
        out += add_to_out1
        out += add_to_out2
        return out
        #return 1.0/(1.0+np.exp(-X))
