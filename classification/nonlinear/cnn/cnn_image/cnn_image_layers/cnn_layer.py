from abc import ABC, abstractmethod
import numpy as np
import timeit
class CNNLayer(ABC):

    def __init__(self, prev_layer, next_layer):
        self._prev_layer = prev_layer
        self._next_layer = next_layer
        self._tot_input_grad = None

    def _set_next_layer(self, next):
        self._next_layer = next

    def _set_prev_layer(self, prev):
        self._prev_layer = prev

    def predict(self, X, inouts = False):
        out = self._transform(X)
        if inouts is True:
            inouts = [X, out]
        elif isinstance(inouts, list):
            inouts.append(out)
        if self._next_layer is None:
            if inouts is not False:
                return out, inouts
            return out
        return self._next_layer.predict(out, inouts = inouts)

    def _set_total_input_grad_old(self, X, layer_out):
        #see if can make this run faster
        if self._next_layer is None:
            self._tot_input_grad = self._input_grad(X, layer_out)
            return None
        next_layer_tot_input_grad = self._next_layer._tot_input_grad
        input_grad = self._input_grad(X, layer_out)
        self._tot_input_grad = np.zeros(next_layer_tot_input_grad.shape[:3] + X.shape, dtype = np.float64)
        for i in range(self._tot_input_grad.shape[0]):
            for j in range(self._tot_input_grad.shape[1]):
                for k in range(self._tot_input_grad.shape[2]):
                    for w in range(self._tot_input_grad.shape[3]):
                        for h in range(self._tot_input_grad.shape[4]):
                            for d in range(self._tot_input_grad.shape[5]):
                                self._tot_input_grad[i,j,k,w,h,d] = np.sum(next_layer_tot_input_grad[i,j,k,:,:,:] * input_grad[:,:,:,w,h,d])


    def _set_total_input_grad(self, X, layer_out):
        #used tensordot to make this faster. Used some assertion testing, but if something goes wrong, the problem may be here
        #see if can make this run faster
        if self._next_layer is None:
            self._tot_input_grad = self._input_grad(X, layer_out)
            return None
        next_layer_tot_input_grad = self._next_layer._tot_input_grad
        input_grad = self._input_grad(X, layer_out)
        test_tot_input_grad = np.tensordot(next_layer_tot_input_grad, input_grad, axes = [(3,4,5), (0,1,2)])
        #self._set_total_input_grad_old(X, layer_out)
        #np.testing.assert_almost_equal(test_tot_input_grad, self._tot_input_grad)
        self._tot_input_grad = test_tot_input_grad


    #called on the last layer of the net. Backprops the net completely, not just this layer. Consumes inouts in the process.
    def backprop_update_gradient(self, inouts, print_times = False):
        #assumes the last element of inouts is the output of this layer, and that the
        #element before that is the input
        if print_times: start_time = timeit.default_timer()
        layer_out = inouts.pop(len(inouts)-1)
        X = inouts[-1]
        self._set_total_input_grad(X, layer_out)
        self._update_param_grads(X, layer_out)
        if print_times:
            print("layer name: " + str(self.__class__.__name__) + ", backprop layer run time: " + str(timeit.default_timer()-start_time))
            if self._prev_layer is None:
                print("-------------------------------------------------------------------")
        if self._prev_layer is not None:
            self._prev_layer.backprop_update_gradient(inouts, print_times = print_times)


    def step_net_params(self, learn_rate):
        self.step_params(learn_rate)
        if self._next_layer is not None:
            self._next_layer.step_net_params(learn_rate)

    #returns a 6d arr grad, where grad[i,j,k,w,h,d] = partial(this layer[i,j,k])/partial(input[w,h,d])
    #uses the input to this layer, or layer out if necessary, to calculate.
    @abstractmethod
    def _input_grad(self, X, layer_out):
        #gradient should be 6 dimensional array
        pass

    #updates the representation of this layer's parameter gradients (if the layer has adjustable paramaters)
    #the gradient descent step uses this representation as the gradient with which it steps the model's weights.
    @abstractmethod
    def _update_param_grads(self, X, layer_out):
        #by chain rule: dLN/dparams = (dLN/dLn)*(dLn/dparams), so just correct multiply
        #self._tot_input_grad by dlayer/dparams (likely uses total differential and summing)
        pass

    #uses whatever the representation of this layer's paramter gradients is to step this layer's parameters, if
    #applicable
    @abstractmethod
    def step_params(self, learn_rate):
        pass


    #returns the result of applying the layer to input X. (Does NOT modify X)
    @abstractmethod
    def _transform(self, X):
        pass
