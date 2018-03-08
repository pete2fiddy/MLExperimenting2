from classification.nonlinear.cnn.cnn_image.cnn_image_layers.cnn_layer import CNNLayer
import toolbox.imageop as imageop
import numpy as np

class SoftmaxLayer(CNNLayer):

    def __init__(self, prev_layer = None, next_layer = None):
        CNNLayer.__init__(self, prev_layer, next_layer)

    #returns a 6d arr grad, where grad[i,j,k,w,h,d] = partial(this layer[i,j,k])/partial(input[w,h,d])
    #uses the input to this layer, or layer out if necessary, to calculate.

    def _input_grad(self, X, layer_out):
        grad = np.zeros(layer_out.shape+X.shape, dtype = X.dtype)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                for k in range(grad.shape[2]):
                    for w in range(grad.shape[3]):
                        for h in range(grad.shape[4]):
                            for d in range(grad.shape[5]):
                                if i == w and j == h and k == d:
                                    grad[i,j,k,w,h,d] = layer_out[i,j,k]*(1-layer_out[w,h,d])
                                else:
                                    grad[i,j,k,w,h,d] = -layer_out[i,j,k]*layer_out[w,h,d]
        return grad

    def _update_param_grads(self, X, layer_out):
        return None#layer has no parameters

    def step_params(self, learn_rate):
        return None#layer has no parameters

    def _transform(self, X):
        e_to_X = np.exp(X)
        return e_to_X/np.sum(e_to_X)
