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
        #is a bottleneck function, so speedups here will speedup net significantly
        #already made attempt to speed up -- likely possible to do more though
        if self._next_layer is None:
            self._tot_input_grad = self._input_grad(X, layer_out)
            return None

        next_layer_tot_input_grad = self._next_layer._tot_input_grad
        print("NEXT LAYER NAME: ", self._next_layer.__class__.__name__)
        print("THIS LAYER NAME: ", self.__class__.__name__)

        self._tot_input_grad = np.zeros(layer_out.shape + next_layer_tot_input_grad.shape[:3], dtype = X.dtype)
        input_gradient = self._input_grad(X, layer_out)
        print("input gradient shape: ", input_gradient.shape)
        print("next layer tot input grad shape: ", next_layer_tot_input_grad.shape)
        for i in range(self._tot_input_grad.shape[0]):
            for j in range(self._tot_input_grad.shape[1]):
                for k in range(self._tot_input_grad.shape[2]):

                    for w in range(input_gradient.shape[3]):
                        for h in range(input_gradient.shape[4]):
                            #make sure this is correct. Is possibly fishy that input_gradient[i,j,k,w,h,:] and next_layer_tot_input_grad[w,h,:]
                            #sometimes have differing shapes. I THINK the intended is for input_gradient[i,j,k,w,h,:] to be a vector,
                            #and for next_layer_tot_input_grad[w,h,:] to be a 4D array.
                            #when next_layer_tot_input_grad[w,h,:] is a vector, is possible is for good reason.

                            #summing in such a manner should be fine given no dimension mismatch errors, and
                            #was previously summing using a for loop over d wherever a ":" is now
                            #Also, when asserted that the modification of this function for speed performed equivalently as previously,
                            #no assertion errors were ever thrown
                            #self._tot_input_grad[i,j,k] += np.sum(input_gradient[i,j,k,w,h,:] * next_layer_tot_input_grad[w,h,:])
                            self._tot_input_grad[i,j,k] += np.sum(next_layer_tot_input_grad[w,h,:] * input_gradient[i,j,k,w,h,:])
        #print("max tot input grad: ", self._tot_input_grad.max())

    def _set_total_input_grad(self, X, layer_out):
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

    def _set_total_input_grad_new(self, X, layer_out):
        #is a bottleneck function, so speedups here will speedup net significantly
        #already made attempt to speed up -- likely possible to do more though
        if self._next_layer is None:
            self._tot_input_grad = self._input_grad(X, layer_out)
            return None

        next_layer_tot_input_grad = self._next_layer._tot_input_grad
        self._tot_input_grad = np.zeros(layer_out.shape + next_layer_tot_input_grad.shape[:3], dtype = X.dtype)
        input_gradient = self._input_grad(X, layer_out)
        #this might be TOTALLY wrong. Note how much paring down I could do compared to _set_total_input_grad_old.
        #error seems to decrease as net trains, though, and this applies to all layers as well.
        #also is odd that _tot_input_grad has to += the right side, and can't = the right side, even though
        #it is initialized to the zero array. LIkely because of the fact that next layer tot input grad can be
        #a non-6D volume sometimes.

        self._tot_input_grad += np.sum(next_layer_tot_input_grad[:,:,:]*input_gradient[:,:,:,:,:,:])#should be multiplying last 3 axises
        #of input_gradient elementise by the first 3 indices of next_layer_tot_input_grad

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
