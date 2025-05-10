import numpy as np
def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """

    N,C,H,W = x.shape
    HH, WW = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]

    h_ = 1+(H-HH)//stride
    w_ = 1+(W-WW)//stride

    out = np.zeros((N,C,h_,w_))

    for n in range(N):
      for c in range(C):
        for hi in range(h_):
          for wi in range(w_):
            vertical = stride*hi
            v_end = vertical+HH

            horizontal = stride*wi
            h_end = horizontal+WW

            x_slice = x[n,c,vertical:v_end,horizontal:h_end]

            out[n,c,hi,wi] = np.max(x_slice)
            
    cache = (x,pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None

    x, pool_param = cache
    N,C,H,W = x.shape
    HH, WW = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]

    h_, w_ = dout.shape[2], dout.shape[3]

    dx = np.zeros(x.shape)

    for n in range(N):
      for c in range(C):
        for hi in range(h_):
          for wi in range(w_):
            vertical = stride*hi
            v_end = vertical+HH

            horizontal = stride*wi
            h_end = horizontal+WW

            x_slice = x[n,c,vertical:v_end,horizontal:h_end]

            mask = (x_slice==np.max(x_slice))

            dx[n,c,vertical:v_end,horizontal:h_end] += dout[n,c,hi,wi] * mask

    return dx 