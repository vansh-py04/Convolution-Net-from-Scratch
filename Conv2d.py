import numpy as np
def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    pad = conv_param["pad"]
    stride = conv_param["stride"]

    # Calculating the spatial dimensions  of the output feature map
    h_ = 1+(H+2*pad-HH)//stride
    w_ = 1+(W+2*pad-WW)//stride

    # output initialization
    out = np.zeros((N,F,h_,w_))
    # padding the input
    x_padded = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode="constant")

    for n in range(N):
      for f in range(F):
        for hi in range(h_):
          for wi in range(w_):
             vertical = stride*hi
             v_end = vertical + HH

             horizontal = stride*wi
             h_end = horizontal + WW

             x_slice = x_padded[n,:,vertical:v_end,horizontal:h_end]

             out[n,f,hi,wi] = np.sum(x_slice*w[f]) + b[f]

    cache = (x,w,b,conv_param)
    return out, cache

def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    x,w,b,conv_param = cache
    N,C,H,W = x.shape
    F,_,HH,WW = w.shape
    pad = conv_param["pad"]
    stride = conv_param["stride"]

    # Retrieving the spatial dimensions of feature map
    h_ = dout.shape[2]
    w_ = dout.shape[3]
    
    # intializing derivatives
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)

    # padding 
    x_padded = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode="constant")
    dx_padded = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode="constant")

    for n in range(N):
      for f in range(F):
        for hi in range(h_):
          for wi in range(w_):
             vertical = stride*hi
             v_end = vertical+HH


             horizontal = stride*wi
             h_end = horizontal+WW

             x_slice = x_padded[n,:,vertical:v_end,horizontal:h_end]

             dw[f] += x_slice*dout[n,f,hi,wi]
             db[f] += dout[n,f,hi,wi]

             dx_padded[n,:,vertical:v_end,horizontal:h_end] += w[f] * dout[n,f,hi,wi]
    
    dx = dx_padded[:,:,pad:-pad,pad:-pad]

    return dx, dw, db