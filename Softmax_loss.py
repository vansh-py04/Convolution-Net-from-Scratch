import numpy as np
def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x"""

    loss = None
    dx = None

    scores = x - np.max(x,axis=1,keepdims=True) # ensures stability by scaling the values
    p = np.exp(scores)
    p /= np.sum(p,axis=1,keepdims=True)
    logp = np.log(p) 
    loss = -logp[np.arange(len(y)),y] # cross entropy of true class
    loss = np.mean(loss) # single scalar loss value representing the average error for the batch

    dx = p
    dx[np.arange(len(y)),y] -= 1 # gradient calculation
    dx /= len(y) # to average the gradient across the batch

    return loss, dx