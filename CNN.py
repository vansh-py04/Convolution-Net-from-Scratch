import numpy as np 
from builtins import object
from Sandwich_layers import *
from Conv2d import *
from Maxpool import *
from ReLU import *
from Affine_layer import *
from Softmax_loss import *

class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # Initializing Weights using Gaussian centered at 0.0
        # with standard deviation equal to weight_scale; 
        # and Biases initialized to zero.
        c,h,w = input_dim
        self.params["W1"] = np.random.randn(num_filters,c,filter_size,filter_size) * weight_scale
        self.params["b1"] = np.zeros(num_filters)

        fc = num_filters*(h//2)*(w//2) 
        self.params["W2"] = np.random.randn(fc,hidden_dim) * weight_scale
        self.params["b2"] = np.zeros(hidden_dim)

        self.params["W3"] = np.random.randn(hidden_dim,num_classes) * weight_scale
        self.params["b3"] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    
    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None

        # forward pass 
        print("Shapes of Output :- ")
        output_conv_relu_max, cache_conv_relu_max = conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
        output_fc_relu, cache_fc_relu = affine_relu_forward(output_conv_relu_max,W2,b2)
        scores, cache_fc = affine_forward(output_fc_relu,W3,b3)

        print("FC output shape:",scores.shape)

        if y is None:
            return scores

        loss, grads = 0, {}

        # backward pass
        loss, dout = softmax_loss(scores,y)
        # simplify the expression for the gradient. 
        loss += self.reg * 0.5 * (np.sum(W1**2) +np.sum(W2**2) +np.sum(W3**2))

        dx3, dw3, db3 = affine_backward(dout,cache_fc)
        dw3 += self.reg * W3 #L2 regularization
        grads["W3"], grads["b3"] = dw3, db3

        dx2, dw2, db2 = affine_relu_backward(dx3,cache_fc_relu)
        dw2 += self.reg * W2
        grads["W2"], grads["b2"] = dw2, db2

        dx1, dw1, db1 = conv_relu_pool_backward(dx2,cache_conv_relu_max)
        dw1 += self.reg * W1
        grads["W1"], grads["b1"] = dw1, db1

        return loss, grads

