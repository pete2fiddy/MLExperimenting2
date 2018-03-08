from classification.nonlinear.cnn.cnn_image.cnn_image_layers.cnn_layer import CNNLayer
import numpy as np

class FullyConnectedLayer(CNNLayer):
    RAND_WEIGHT_RANGE = (-.5,.5)#range needs to be small, esp with act functions like tanh and sigmoid because there are a lot
    BIAS_LEARN_RATE_MULTIPLIER = 1
    #of inputs, so mildly large values will create dot products that yield a very high value, meaning tanh or sigmoid will have a
    #value very close to 0 or 1, and will therefore have a tiny gradient
    def __init__(self, num_nodes, input_dims, prev_layer = None, next_layer = None):
        CNNLayer.__init__(self, prev_layer, next_layer)
        #represents weights as a 6D array for compatability with the rest of the net
        self.__input_dims = input_dims
        #where self.__weights[i,w,h,d] is the weight connecting input [w,h,d] to node i
        self.__weights = FullyConnectedLayer.RAND_WEIGHT_RANGE[0]+(FullyConnectedLayer.RAND_WEIGHT_RANGE[1]-\
        FullyConnectedLayer.RAND_WEIGHT_RANGE[0])*np.random.rand(num_nodes, input_dims[0], input_dims[1], input_dims[2])
        self.__weights_grad = np.zeros(self.__weights.shape, dtype = np.float64)
        self.__biases = FullyConnectedLayer.RAND_WEIGHT_RANGE[0]+(FullyConnectedLayer.RAND_WEIGHT_RANGE[1]-\
        FullyConnectedLayer.RAND_WEIGHT_RANGE[0])*np.random.rand(num_nodes)
        self.__biases_grad = np.zeros(self.__biases.shape, dtype = np.float64)

    def _input_grad(self, X, layer_out):
        grad = np.zeros(layer_out.shape + X.shape, dtype = np.float64)
        for i in range(grad.shape[0]):
            grad[i,0,0] = self.__weights[i,:,:,:]
        return grad

    def _update_param_grads_old(self, X, layer_out):
        #by chain rule: dLN/dparams = (dLN/dLn)*(dLn/dparams), so just correct multiply
        #self._tot_input_grad by dlayer/dparams (likely uses total differential and summing)
        part_weights_grad = self.__calc_weights_grad(X, layer_out)
        #part_biases_grad = self.__calc_biases_grad(X, layer_out)
        #print("max self.input_grads: ", (self._tot_input_grad**2).max())
        for w1 in range(self._tot_input_grad.shape[0]):
            for h1 in range(self._tot_input_grad.shape[1]):
                for d1 in range(self._tot_input_grad.shape[2]):
                    for w2 in range(self._tot_input_grad.shape[3]):
                        for h2 in range(self._tot_input_grad.shape[4]):
                            for d2 in range(self._tot_input_grad.shape[5]):
                                #can likely remove h2 and d2 since it is iterative over padded dimensions -- can just set
                                #their value to a constant 0
                                self.__weights_grad += self._tot_input_grad[w1,h1,d1,w2,h2,d2]*part_weights_grad[w2]
                                #print("amount added: ", self._tot_input_grad[w1,h1,d1,w2,h2,d2]*part_weights_grad[w2])
                                #print("weights: ", self.__weights)
                                self.__biases_grad[w2] += self._tot_input_grad[w1,h1,d1,w2,h2,d2]
        '''
        print("self.biases grad.shape: ", self.__biases_grad.shape)
        for i in range(self.__biases_grad.shape[0]):
            self.__biases_grad[i] += self._tot_input_grad[]
        self.__biases_grad += np.sum(self._tot_input_grad, axis = 3)
        '''

    def _update_param_grads(self, X, layer_out):
        #by chain rule: dLN/dparams = (dLN/dLn)*(dLn/dparams), so just correct multiply
        #self._tot_input_grad by dlayer/dparams (likely uses total differential and summing)
        part_weights_grad = self.__calc_weights_grad(X, layer_out)
        #part_biases_grad = self.__calc_biases_grad(X, layer_out)
        #print("max self.input_grads: ", (self._tot_input_grad**2).max())
        for w1 in range(self._next_layer._tot_input_grad.shape[0]):
            for h1 in range(self._next_layer._tot_input_grad.shape[1]):
                for d1 in range(self._next_layer._tot_input_grad.shape[2]):
                    for w2 in range(self._next_layer._tot_input_grad.shape[3]):
                        for h2 in range(self._next_layer._tot_input_grad.shape[4]):
                            for d2 in range(self._next_layer._tot_input_grad.shape[5]):
                                #can likely remove h2 and d2 since it is iterative over padded dimensions -- can just set
                                #their value to a constant 0
                                self.__weights_grad += self._next_layer._tot_input_grad[w1,h1,d1,w2,h2,d2]*part_weights_grad[w2]
                                #print("amount added: ", self._tot_input_grad[w1,h1,d1,w2,h2,d2]*part_weights_grad[w2])
                                #print("weights: ", self.__weights)
                                self.__biases_grad[w2] += self._next_layer._tot_input_grad[w1,h1,d1,w2,h2,d2]

        '''
        print("self.biases grad.shape: ", self.__biases_grad.shape)
        for i in range(self.__biases_grad.shape[0]):
            self.__biases_grad[i] += self._tot_input_grad[]
        self.__biases_grad += np.sum(self._tot_input_grad, axis = 3)
        '''

    def __calc_weights_grad(self, X, layer_out):
        #returns the partial derivative of the ith node of the layer w.r.t. self.__weights[i,w,h,d] of X. (derivative is zero if
        #first axis of weights is not the same as the node whose partial derivative is being taken)
        #since is internal, will just use the 4d representation because the output does not need to be represented as a volume of 6 dimensions.
        #(4d representation in that the output array will index into partial(node i)/partial())
        grad = np.zeros((layer_out.shape[0],) + X.shape, dtype = np.float64)
        for i in range(grad.shape[0]):
            grad[i] = X
        return grad

    '''
    def __calc_biases_grad(self, X, layer_out):
        #returns the partial derivative of the ith node of the layer w.r.t. self.__biases[i] (not 2d gradient because if index of bias
        #!= index of node, derivative is zero)
        return np.ones(self.__biases.shape, dtype = np.float64)
    '''


    def step_params(self, learn_rate):

        #code here
        self.__weights += learn_rate * self.__weights_grad
        self.__biases += FullyConnectedLayer.BIAS_LEARN_RATE_MULTIPLIER * learn_rate * self.__biases_grad
        self.__weights_grad = np.zeros(self.__weights_grad.shape, dtype = np.float64)
        self.__biases_grad = np.zeros(self.__biases_grad.shape, dtype = np.float64)


    def _transform(self, X):
        out = np.zeros((self.__biases.shape[0], 1, 1), dtype = np.float64)
        for i in range(out.shape[0]):
            out[i, 0, 0] = np.sum(self.__weights[i,:,:,:]*X[:,:,:]) + self.__biases[i]
        return out
