import numpy as np

def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    
    return (np.matmul(X, W) + b).tolist() # (n, out_dim), (_, out_dim) so broadcasting works