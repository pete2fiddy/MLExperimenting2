import numpy as np

#returns the output of convolving image img with the kernel,with stride stride(stride_rows, stride_cols), and zero padding
#zero_padding (num rows to pad on one side, num cols to pad on one side)
def convolve(img, kernel, stride = (1,1), zero_padding = (0,0)):
    #try to make this faster if possible
    half_kernel0 = int((kernel.shape[0]-1)/2)
    half_kernel1 = int((kernel.shape[1]-1)/2)
    conv_cols = int((2*zero_padding[0]+img.shape[0]-2*half_kernel0)/(stride[0]))
    conv_rows = int((2*zero_padding[1]+img.shape[1]-2*half_kernel1)/(stride[1]))
    img = zero_pad(img, zero_padding)
    conv = np.zeros((conv_cols, conv_rows), dtype = img.dtype)
    conv_i = 0
    conv_j = 0
    for i in range(half_kernel0,  img.shape[0]-kernel.shape[0]+half_kernel0, stride[0]):
        for j in range(half_kernel1, img.shape[1]-kernel.shape[1]+half_kernel1, stride[1]):
            i_minus_half_kernel0 = i-half_kernel0
            j_minus_half_kernel1 = j-half_kernel1

            img_window = img[i_minus_half_kernel0:i_minus_half_kernel0+kernel.shape[0],\
            j_minus_half_kernel1:j_minus_half_kernel1+kernel.shape[1]]
            conv[conv_i, conv_j] = np.sum(kernel*img_window)
            conv_j += 1
        conv_j = 0
        conv_i += 1
    return conv

def zero_pad(img, zero_padding):
    zero_padded = np.zeros((img.shape[0] + zero_padding[0]*2, img.shape[1] + zero_padding[1]*2), dtype = img.dtype)
    zero_padded[zero_padding[0]:zero_padding[0]+img.shape[0], zero_padding[1]:zero_padding[1]+img.shape[1]] = img
    return zero_padded

def maxpool(img, kernel_size, stride):
    pool = np.zeros((int((img.shape[0] - kernel_size[0])/stride[0]), int((img.shape[1]-kernel_size[1])/stride[1])))
    for i in range(0, pool.shape[0]):
        for j in range(0, pool.shape[1]):
            row_bounds = (stride[0]*kernel_size[0], stride[0]*(kernel_size[0]+1))
            col_bounds = (stride[1]*kernel_size[1], stride[1]*(kernel_size[1]+1))
            pool[i,j] = img[row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1]].max()
    return pool
