from classification.nonlinear.cnn.cnn_image.cnn_image_layers.cnn_layer import CNNLayer
import toolbox.imageop as imageop
import numpy as np

class SumSquareErrorLayer(CNNLayer):

    def __init__(self, prev_layer = None):
        CNNLayer.__init__(self, prev_layer, None)

    #returns a 6d arr grad, where grad[i,j,k,w,h,d] = partial(this layer[i,j,k])/partial(input[w,h,d])
    #uses the input to this layer, or layer out if necessary, to calculate.
    #Note: in this case, layer_out = the target value for X (y)
    def _input_grad(self, X, layer_out):
        assert np.not_equal(X,layer_out).any()#to make sure I'm not funneling in the output of this layer
        #make this faster.
        grad = np.zeros(layer_out.shape + X.shape, dtype = np.float64)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                for k in range(grad.shape[2]):
                    grad[i,j,k,i,j,k] = (X[i,j,k]-layer_out[i,j,k])
        return grad

    #updates the representation of this layer's parameter gradients (if the layer has adjustable paramaters)
    #the gradient descent step uses this representation as the gradient with which it steps the model's weights.
    def _update_param_grads(self, X, layer_out):
        #by chain rule: dLN/dparams = (dLN/dLn)*(dLn/dparams), so just correct multiply
        #self._tot_input_grad by dlayer/dparams (likely uses total differential and summing)
        return None#layer has no parameters

    #uses whatever the representation of this layer's paramter gradients is to step this layer's parameters, if
    #applicable
    def step_params(self, learn_rate):
        return None#layer has no parameters


    #returns the result of applying the layer to input X. (Does NOT modify X)
    def _transform(self, X):
        return X

    def error(self, predict, y):
        return 0.5*np.sum(np.square(y - predict))
