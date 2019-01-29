from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ResNet(object):
    """
    A resnet convolutional network with the following architecture:

    For cifar 10 data, architecture is simple:
    3x3conv - residual_block* -avgpool -fc10-softmax
        
    For image net data:
    [conv-bn-relu-pool] - residual_block*sum(list_layers) - avgpool - softmax
    
    * call it large layers as "stage"
    * each residual_block has 2 layers and use same number of convparam
    * e.g. resnet18 list_layers = [2,2,2,2], num_total_layers = 1+2*8+1 = 18
    
    Residual_block:
    x -   [conv-bn-relu]-[conv-bn-]        + relu
      |                                    | 
       -- Identity(optinal downsampling) --
       
    Down sampling is used so that original input can be extended to the output 
    of [conv-bn-relu]-[conv-bn], it only happens in convX_1. That's also the 
    reason that pytorch has the _make_layer function, makes that down_sampling 
    only happen once in each of the 4 big section.
    Downsampling itself is a 1x1 convolution(to extend to the correct number of
    channels) + bn
    Some the size of the image is only affected by downsampling, the conv3x3 
    wont't affect image size since H'= 1 + (H + 2 * pad - HH) / stride == H using
    HH=3, pad=1, stride=1
    
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=[3, 32, 32], layers = [1], num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - layers: how many resisual blocks in each stage
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.conf = {}
        self.reg = reg
        self.dtype = dtype
        self.layers = layers

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        
        ##########################
        #   preparation stage    #
        ##########################
        
        C, H, W = input_dim # C, H, W will track output size in current layer
        
        # this convolution doesn't change h and w
        conv_param = {'stride': 1, 'pad': 1}
        W0 = np.random.normal(loc=0.0, scale = weight_scale, size = (16, C, 3, 3))
        b0 = np.zeros(16)
        C, H, W = 16, H, W
        self.params['W0'] = W0
        self.params['b0'] = b0        
        self.conf['conv_param'] = conv_param
        
        ##########################
        #   main stages          #
        ##########################
        

        
        assert len(layers) ==1, 'more than 1 stage is now supported'
        
        # the Weights are names as W%(stage_name)_%(residual_block_index)_%(convolution_layer)
        for stage_id in np.arange(1, len(layers) + 1):
            num_residual_blks = layers[stage_id -1]

            # might need downsampling
            # some operation
            # C. H, W = xxx
            
            for blk_id in np.arange(num_residual_blks):
                WW1 = np.random.normal(loc=0.0, scale = weight_scale, size = (16, C,3,3))
                bb1 = np.zeros(16)
                WW2 = np.random.normal(loc=0.0, scale = weight_scale, size = (16, C,3,3))
                bb2 = np.zeros(16)
            
                self.params['W' + str(stage_id) + '_' + str(blk_id) + '_0' ] = WW1
                self.params['b' + str(stage_id) + '_' + str(blk_id) + '_0'] = bb1
                self.params['W' + str(stage_id) + '_' + str(blk_id) + '_1' ] = WW2
                self.params['b' + str(stage_id) + '_' + str(blk_id) + '_1'] = bb2
                
        

        ##########################
        #   final stages         #
        ##########################
        
        # global pool TODO: use avg pool instead of max poo,
        pool_param = {'pool_height': H, 'pool_width': W, 'stride': 1}
        self.conf['pool_param'] = pool_param
        
        # fc-10
        W3 = np.random.normal(loc=0.0, scale = weight_scale, size = (C, num_classes))
        b3 = np.zeros(num_classes)
        self.params['W3'] = W3
        self.params['b3'] = b3
        

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        conv_param = self.conf['conv_param']
        pool_param = self.conf['pool_param'] 
        layers = self.layers
        
        all_caches = []

        
        ##########################
        #   preparation stage    #
        ##########################
        W0, b0 = self.params['W0'], self.params['b0']
        out, cache = conv_relu_forward(X, W0, b0, conv_param)
        all_caches.append(cache)
        
        
        ##########################
        #   main stages          #
        ##########################
        for stage_id in np.arange(1, len(layers) + 1): # 1,...n-1, n
            num_residual_blks = layers[stage_id -1]
            # might need downsampling
            # some operation
            # C. H, W = xxx
            
            for blk_id in np.arange(num_residual_blks):
                # print('----forwarding at stage %d blk %d' %( stage_id, blk_id))
                # TODO: change to ref2W
                WW1 = self.params['W' + str(stage_id) + '_' + str(blk_id) + '_0' ]
                bb1 = self.params['b' + str(stage_id) + '_' + str(blk_id) + '_0']
                WW2 = self.params['W' + str(stage_id) + '_' + str(blk_id) + '_1' ]
                bb2 = self.params['b' + str(stage_id) + '_' + str(blk_id) + '_1']
                
                out, cache = resnet_basic_no_bn_forward(out, WW1, bb1, WW2, bb2, conv_param)
                all_caches.append(cache)
        
        ##########################
        #   final stages         #
        ##########################
        
        # golobal pool
        out, cache =  max_pool_forward_fast(out, pool_param)
        all_caches.append(cache)
            
        # fc-10
        W3, b3 = self.params['W3'], self.params['b3']
        out, cache = affine_forward(out, W3, b3)
        all_caches.append(cache)
            
        scores = out

       
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the residual  net,                 #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        reg = self.reg
        loss, dout = softmax_loss(scores, y)
        
        ###############
        # Final Stage #
        ###############
        loss += 0.5*reg*np.sum(np.square(W3))
        
        dout, dW3, db3 = affine_backward(dout, all_caches.pop())
        dout = max_pool_backward_fast(dout, all_caches.pop())
        
        dW3 += self.reg*W3
        
        grads['W3'] = dW3
        grads['b3'] = db3
        
        
        ###############
        # Main Stage  #
        ###############
        for stage_id in np.arange(len(layers), 0, -1): # n, n-1, 1
            num_residual_blks = layers[stage_id -1]

            for blk_id in np.arange(num_residual_blks-1, -1, -1): # blk n-1... blk 0
                # print('---backprog at stage %d blk %d' %( stage_id, blk_id))
                WW1 = self.params['W' + str(stage_id) + '_' + str(blk_id) + '_0' ]
                WW2 = self.params['W' + str(stage_id) + '_' + str(blk_id) + '_1' ]
        
                # this should be in the forward pass
                loss += 0.5*reg*(np.sum(np.square(WW1)) + np.sum(np.square(WW2)))
                
                dout, dWW1, dbb1, dWW2, dbb2 = resnet_basic_no_bn_backward(dout, all_caches.pop())
                
                
                dWW1 += self.reg*WW1 # derivative of regulaizer
                dWW2 += self.reg*WW2
                
                grads['W' + str(stage_id) + '_' + str(blk_id) + '_0'] = dWW1
                grads['b' + str(stage_id) + '_' + str(blk_id) + '_0'] = dbb1
                grads['W' + str(stage_id) + '_' + str(blk_id) + '_1'] = dWW2
                grads['b' + str(stage_id) + '_' + str(blk_id) + '_1'] = dbb2
                         
                
        ###############
        # Prepare Stage#
        ###############
        loss += 0.5*reg*np.sum(np.square(W0))
        
        dx, dW0, db0 = conv_relu_backward(dout, all_caches.pop())
        
        dW0 += self.reg*W0
           
        grads['W0'] = dW0
        grads['b0'] = db0
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads



class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
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

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        # * why is this called 3 layers?  
        #   This means  conv/fc layers are counted because they have weights!
        #   layer1 conv - relu - 2x2 max pool
        #   layer2 affine - relu
        #   layer3 softmax
        # * conv-relu-pool 
        # 0. (N, C=3, H=32, W=32)
        # 1. conv,  (N=3,F=32,H' = 28,W') H' = 1 + (H + 2 * pad - HH) / stride
        # 2. relu (N,F, H',W')
        # 3. 2x2 max pool (N,F=32, H''=14,W''), H'' = 1 + (H' - pool_height) / stride
        
        W1 = np.random.normal(loc=0.0, scale = weight_scale, size = (num_filters, input_dim[0], filter_size, filter_size))
        b1 = np.zeros(num_filters)
        # after  conv-relu-pool
        
        # conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        out_dim= input_dim[1] # preserved in loss function TODO: makes it intelligient
        out_dim = out_dim//2 # pool will 
        C, H, W = num_filters,  out_dim, out_dim 
        self.params['W1'] = W1
        self.params['b1'] = b1
        
        # affine1
        W2 = np.random.normal(loc=0.0, scale = weight_scale, size = (C*H*W, hidden_dim))
        b2 = np.zeros(hidden_dim)
        self.params['W2'] = W2
        self.params['b2'] = b2
        
        #affine2
        W3 =  np.random.normal(loc=0.0, scale = weight_scale, size = (hidden_dim, num_classes))
        b3 =  np.zeros(num_classes)
        self.params['W3'] = W3
        self.params['b3'] = b3
        

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        #  layer1 conv - relu - 2x2 max pool
        #   layer2 affine - relu
        #   layer3 affine
        out, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out, cache2 = affine_relu_forward(out, W2, b2)
        scores, cache3 = affine_forward(out, W3, b3)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        reg = self.reg
        loss, dout = softmax_loss(scores, y) 
        loss += 0.5*reg*(np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))

        dout, dW3, db3 = affine_backward(dout, cache3)
        dout, dW2, db2 =affine_relu_backward(dout, cache2)
        dx, dW1, db1 = conv_relu_pool_backward(dout, cache1)
        
        dW1 += self.reg*W1
        dW2 += self.reg*W2
        dW3 += self.reg*W3
        
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W3'] = dW3
        grads['b3'] = db3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
