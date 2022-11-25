import numpy as np

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e**2)

def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return calculate_mse(e)

def compute_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def gradient_descent(y, tx, initial_w, max_iters, gamma):

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    prev_loss = 0
    loss = 0
    w = initial_w
    
    for n_iter in range(max_iters):
        ### SOLUTION
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        prev_loss = loss
        loss = calculate_mse(err)
        
        if loss < prev_loss:
            w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        
        print("Iteration:", n_iter)
        print("Loss:", loss)
        print("Prev loss:", prev_loss)

    return w, loss

def standardize(x):
    #Standardize the original data set.
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x

def least_squares(y, tx):
    #We solve the normal equations using QR decomposition, which is a computationally efficient method.
    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    w = np.linalg.solve(a,b)

    err = y - tx.dot(w)
    loss = np.mean(err**2)/2
    
    return w, loss

def predict(tx, w):
    """ 
    Computes the age prediction according to the weight matrix.
    The return value is rounded to the nearest integer. 
    """
    y = tx.dot(w)
    return np.rint(y)

def shuffle_data(y, tx, seed=1):
    np.random.seed(seed)
    inds = np.random.permutation(tx.shape[0])
    tx = tx[inds,:]
    y = y[inds]
    return y, tx

# Slice data and labels into training and validation sets
def slice_data(y, tx, ratio, seed=1): 
    slice_id = int(np.floor(y.shape[0]*ratio))
    y_va, y_tr = y[:slice_id], y[slice_id:]
    tx_va, tx_tr = tx[:slice_id,:], tx[slice_id:,:]
    return y_va, y_tr, tx_va, tx_tr

def accuracy(a, b):
    return np.sum(a == b)/a.shape[0]