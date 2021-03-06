from classification.nonlinear.cnn.cnn_image.cnn_image_layers.cnn_layer import CNNLayer
import toolbox.imageop as imageop
import numpy as np

class ConvolutionLayer(CNNLayer):
    RAND_WEIGHT_RANGE = (-2, 2)
    BIAS_LEARN_RATE_MULTIPLIER = .01

    def __init__(self, kernel_dim, num_kernels, stride, zero_padding, prev_layer = None, next_layer = None):
        CNNLayer.__init__(self, prev_layer, next_layer)



        self.__half_kernel0 = int((kernel_dim[0]-1)/2)
        self.__half_kernel1 = int((kernel_dim[1]-1)/2)

        #self.__kernels = np.random.normal(scale = .01, size = kernel_dim + (num_kernels,))
        #self.__bias = np.random.normal(scale = .01, size = 1)[0]

        #for testing:
        self.__kernels = np.zeros(kernel_dim + (num_kernels,))
        self.__kernels[self.__half_kernel0, self.__half_kernel1, :] = 1
        self.__kernels += np.random.normal(scale = .01, size = kernel_dim + (num_kernels,))
        self.__bias = 0


        self.__stride = stride
        self.__zero_padding = zero_padding
        self.__bias_grad = 0
        self.__kernels_grad = np.zeros(self.__kernels.shape, dtype = np.float64)

    def _input_grad(self, X, layer_out):
        #already made attempt to speed up -- likely possible to do more though
        grad = np.zeros(layer_out.shape + X.shape, dtype = X.dtype)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                for k in range(grad.shape[2]):
                    for kernel_w in range(self.__kernels.shape[0]):
                        for kernel_h in range(self.__kernels.shape[1]):
                            w = kernel_w + self.__stride[0]*i - self.__half_kernel0
                            h = kernel_h + self.__stride[1]*j - self.__half_kernel1
                            if w >= 0 and w < grad.shape[3] and h >= 0  and h < grad.shape[4]:
                                grad[i,j,k,w,h,:] = self.__kernels[kernel_w, kernel_h, k]
        return grad

    def __kernel_grad(self, X):
        #already made attempt to speed up -- likely possible to do more though
        X_zero_padded = self.__zero_pad_X(X)
        grad = np.zeros(self.__transform_shape(X) + self.__kernels.shape, dtype = self.__kernels.dtype)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                X_i = int(i*self.__stride[0]-self.__half_kernel0)
                X_j = int(j*self.__stride[1]-self.__half_kernel1)
                for k in range(grad.shape[2]):
                    for w in range(grad.shape[3]):
                        for h in range(grad.shape[4]):
                            grad[i,j,k,w,h,k] = np.sum(X_zero_padded[X_i+w, X_j+h, :])
        return grad

    '''
    def _update_param_grads_old1(self, X, layer_out):
        kernel_grad = self.__kernel_grad(X)
        for w1 in range(self._tot_input_grad.shape[0]):
            for h1 in range(self._tot_input_grad.shape[1]):
                for d1 in range(self._tot_input_grad.shape[2]):
                    for w2 in range(self._tot_input_grad.shape[3]):
                        for h2 in range(self._tot_input_grad.shape[4]):
                            for d2 in range(self._tot_input_grad.shape[5]):
                                self.__kernels_grad += self._tot_input_grad[w1,h1,d1,w2,h2,d2]*kernel_grad[w2,h2,d2,:,:,:]
        self.__bias_grad += np.sum(self._tot_input_grad)

    def _update_param_grads_old2(self, X, layer_out):
        #assuming this is equivalent to _update_param_grads_old, because the difference between the old kernel and this one
        #is a very small number. Likely minor under/overflow/lossiness occurring
        print("self._tot_input_grad.shape: ", self._tot_input_grad.shape)
        kernel_grad = self.__kernel_grad(X)
        for w2 in range(self._tot_input_grad.shape[3]):
            for h2 in range(self._tot_input_grad.shape[4]):
                for d2 in range(self._tot_input_grad.shape[5]):
                    self.__kernels_grad += np.sum(self._tot_input_grad[:,:,:,w2,h2,d2])*kernel_grad[w2,h2,d2,:,:,:]
        self.__bias_grad += np.sum(self._tot_input_grad)
        #print("difference sum: ", np.sum(np.square(self.__kernels_grad - old_kernels_grad)))
    '''

    def _update_param_grads_old(self, X, layer_output):
        kernel_grad = self.__kernel_grad(X)
        #sum over next layer's total input derivative, times
        #sum over indices of denominator of dlN/dlthislayer(which is total input derivative of next layer) and of numerator of
        #dlThisLayer/dweight[w,h,d]
        kernels_grad_add = np.zeros(self.__kernels_grad.shape, dtype = np.float64)
        next_layer_tot_grad = self._next_layer._tot_input_grad
        for i in range(self.__kernels_grad.shape[0]):
            for j in range(self.__kernels_grad.shape[1]):
                for k in range(self.__kernels_grad.shape[2]):
                    #sums over all contributions to the cost, over all inputs of this layer, w.r.t. the element of kernels [i,j,k]
                    #kernels_grad_add[i,j,k] = np.sum(self._next_layer._tot_input_grad * kernel_grad[:,:,:,i,j,k])
                    for w in range(next_layer_tot_grad.shape[0]):
                        for h in range(next_layer_tot_grad.shape[1]):
                            for d in range(next_layer_tot_grad.shape[2]):
                                kernels_grad_add[i,j,k] += np.sum(next_layer_tot_grad[w,h,d,:,:,:]*kernel_grad[:,:,:,i,j,k])

        self.__kernels_grad += kernels_grad_add
        self.__bias_grad += np.sum(self._next_layer._tot_input_grad)

    def _update_param_grads(self, X, layer_output):
        kernel_grad = self.__kernel_grad(X)
        #sum over next layer's total input derivative, times
        #sum over indices of denominator of dlN/dlthislayer(which is total input derivative of next layer) and of numerator of
        #dlThisLayer/dweight[w,h,d]

        #used tensordot to make this faster. Used some assertion testing, but if something goes wrong, the problem may be here
        #see if can make this run faster
        next_layer_tot_grad = self._next_layer._tot_input_grad

        final_layer_kernels_grad = np.tensordot(next_layer_tot_grad, kernel_grad, axes = [(3,4,5), (0,1,2)])
        test_kernels_grad = np.sum(final_layer_kernels_grad, axis = (0,1,2))

        #self._update_param_grads_old(X, layer_output)
        #np.testing.assert_almost_equal(test_kernels_grad, self.__kernels_grad)
        #self.__kernels_grad += kernels_grad_add
        self.__kernels_grad += test_kernels_grad
        self.__bias_grad += np.sum(self._next_layer._tot_input_grad)



    def step_params(self, learn_rate):
        #for testing
        #return None

        #print("mean kernel magnitude: ", np.average(np.abs(self.__kernels)))
        #print("mean kernel gradient magnitude: ", np.average(np.abs(self.__kernels_grad)))
        #print("CONV bias magnitude: ", abs(self.__bias))
        #print("CONV bias gradient magnitude: ", abs(self.__bias_grad))
        kernel_grad_mag = np.linalg.norm(learn_rate*self.__kernels_grad.flatten())
        bias_grad_mag = abs(learn_rate*ConvolutionLayer.BIAS_LEARN_RATE_MULTIPLIER*self.__bias_grad)
        #print("kernel grad/param ratio: ", kernel_grad_mag/np.linalg.norm(self.__kernels.flatten()))
        #print("bias grad/param ratio: ", bias_grad_mag/abs(self.__bias))
        self.__kernels += learn_rate*self.__kernels_grad
        self.__kernels_grad = np.zeros(self.__kernels_grad.shape, dtype = self.__kernels_grad.dtype)
        self.__bias += ConvolutionLayer.BIAS_LEARN_RATE_MULTIPLIER*learn_rate*self.__bias_grad
        self.__bias_grad = 0



    def _transform(self, X):
        out = np.zeros(self.__transform_shape(X), dtype = X.dtype)
        for k in range(out.shape[2]):
            for d in range(X.shape[2]):
                out[:,:,k] += imageop.convolve(X[:,:,d], self.__kernels[:,:,k], self.__stride, self.__zero_padding)
        out += self.__bias
        return out

    def __transform_shape(self, X):
        out_shape0 = int((X.shape[0]-2*self.__half_kernel0+2*self.__zero_padding[0])/self.__stride[0])
        out_shape1 = int((X.shape[1]-2*self.__half_kernel1+2*self.__zero_padding[1])/self.__stride[1])
        return (out_shape0, out_shape1, self.__kernels.shape[2])

    def __zero_pad_X(self, X):
        padded = np.zeros((X.shape[0]+2*self.__zero_padding[0], X.shape[1]+2*self.__zero_padding[1], X.shape[2]))
        for d in range(padded.shape[2]):
            padded[:,:,d] = imageop.zero_pad(X[:,:,d], self.__zero_padding)
        return padded
