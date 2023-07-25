iimport numpy as np

def mean_squared_error(y_pred, y_true):
    # Check if the shapes of y_pred and y_true are the same
    assert y_pred.shape == y_true.shape, "Shapes of y_pred and y_true must be the same."
    # Get the number of samples
    n = y_pred.shape[0]
    # Compute the squared differences between predicted and true values
    squared_diff = (y_pred - y_true) ** 2
    # Calculate the mean squared error
    mse = np.sum(squared_diff) / n
    return mse

