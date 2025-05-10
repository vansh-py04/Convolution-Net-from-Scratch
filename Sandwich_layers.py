from Conv2d import *
from Maxpool import *
from ReLU import *
from Affine_layer import *
from Softmax_loss import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, affine_cache = affine_forward(x,w,b)
    out, r_cache = relu_forward(a)

    # Shape of feature maps
    print("FC output shape:", a.shape)
    print("ReLU output shape:", out.shape)

    cache = (affine_cache,r_cache)
    return out,cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    affine_cache, r_cache = cache
    dx_relu = relu_backward(dout,r_cache)

    dx,dw, db = affine_backward(dx_relu,affine_cache)

    return dx,dw,db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    conv_out, conv_cache = conv_forward_naive(x,w,b,conv_param)

    r, r_cache = relu_forward(conv_out)

    out, pool_cache = max_pool_forward_naive(r,pool_param)

    # Shape of feature maps
    print("Conv output shape:", conv_out.shape)
    print("ReLU output shape:", r.shape)
    print("Max pool output shape:", out.shape)

    return out, (conv_cache,r_cache,pool_cache)


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, r_cache, pool_cache = cache

    dx_pool = max_pool_backward_naive(dout,pool_cache)

    dx_relu = relu_backward(dx_pool,r_cache)

    dx, dw, db = conv_backward_naive(dx_relu,conv_cache)

    return dx, dw, db