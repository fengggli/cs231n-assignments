pass
from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta

def plus_forward(x1, x2):
    out = x1+x2
    cache = (x1, x2)
    return out, cache

def plus_backward(dout, cache):
    x1, x2 = cache
    dx1 = dout*x2
    dx2 = dout*x1
    return dx1, dx2


# iden is the indentify from below layer
def conv_iden_relu_forward(x, iden, w, b , conv_param):
    out, conv_cache = conv_forward_fast(x, w, b, conv_param)
    #  an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, plus_cache = plus_forward(out, iden)
    out, relu_cache = relu_forward(out)
    cache = (conv_cache, plus_cache, relu_cache)
    return out, cache

def conv_iden_relu_backward(dout, cache):
    conv_cache, plus_cache, relu_cache = cache
    
    dout = relu_backward(dout, relu_cache)
    dout, diden = plus_backward(dout, plus_cache)
    dx, dw, db = conv_backward_fast(dout, conv_cache)
    return dx, diden, dw, db

# TODO: add batch norm
def resnet_basic_no_bn_forward(x, W1, b1, W2, b2, conv_param):
    out, cache1 = conv_relu_forward(x, W1, b1, conv_param)
    out, cache2 = conv_iden_relu_forward(out, iden, W2, b2, conv_param)
    cache = (cache1, cache2)
    return out, cache

def resnet_basic_no_bn_backward(dout, cache):
    cache1, cache2 = cache
    dout1, dout2, diden, dW2, db2 = conv_iden_relu_backward(dout, cache2)
    dout1, dW1, db1 = conv_relu_backward(dout1, cache1)
    dx = dout2 + dout1
    return dx, dW1, db1, dW2, db2


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
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    #print('conv_relu_pool, input' + str(x.shape) + 'output' + str(out.shape))
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
