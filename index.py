import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

### Functions
def jwb(X, y, w, b):
    """
    Calculates squared loss of Function Fwb(x) => J(w,b)

     Args:
      X (ndarray (m,n,))  : Data x, m datasets, n features
      y (ndarray (m,))  : Real y, m datasets
      w (ndarray(n,)): Weight for features n
      b (scalar): Model bias
      
    Returns:
      cost (float) : Squared Cost
    """
    cost = 0
    m = X.shape[0]

    for i in range(m):
        f_wb = np.dot(w, X[i]) + b
        cost += (f_wb - y[i])**2
    
    return cost / (2*m)

def dw_jwb(X, y, w, b):
    """
    Calculates cost of partial derivative w of J(w,b)

    Args:
      X (ndarray (m,n,))  : Data x, m datasets, n features
      y (ndarray (m,))  : Real y, m datasets
      w (ndarray(n,)): Weight for features n
      b (scalar): Model bias
      
    Returns:
      cost (ndarray(n,)) : Cost of partial derivative w of J(w,b) for n features
    """
    m = X.shape[0]
    n = X.shape[1]
    cost = np.zeros((n))

    for i in range(m):
        f_wb = np.dot(w, X[i]) + b
        cost += (f_wb - y[i]) * X[i]
    
    return cost / m

def db_jwb(X, y, w, b):
    """
    Calculates cost of partial derivative b of J(w,b)

    Args:
      X (ndarray (m,n,))  : Data x, m datasets, n features
      y (ndarray (m,))  : Real y, m datasets
      w (ndarray(n,)): Weight for features n
      b (scalar): Model bias
      
    Returns:
      cost (scalar) : Cost of partial derivative b of J(w,b)
    """
    m = X.shape[0]
    cost = 0

    for i in range(m):
        f_wb = np.dot(w, X[i]) + b
        cost += (f_wb - y[i])
    
    return cost / m

def gradient_descent(X, y, w, b, a):
    """
    Executes Gradient Descent and returns updated values model parameters
    
    Args:
      X (ndarray (m,n,))  : Data x, m datasets, n features
      y (ndarray (m,))  : Real y, m datasets
      w (ndarray(n,)): Initial model weight
      b (scalar): initial model bias
      alpha (float): Learning rate
      
    Returns:
      Tuple(w, b):
        w (ndarray(n,)): Updated weight
        b (scalar): Updated bias
    """
    w_new = w - a * dw_jwb(X, y, w, b)
    b_new = b - a * db_jwb(X, y, w, b)

    return w_new, b_new

def predict(x, w, b):
    """
    Predicts y for linear regression model (normalizes input)

     Args:
      x (ndarray (n,))  : Data x, n features
      w (ndarray(n,)): Model weight
      b (scalar): Model bias
      
    Returns:
      y_hat (scalar): Prediction of y
    """
    return np.dot(w, x) + b

def z_score_normalize(X):
    """
    Normalizes features of X to be around [-1, 1] and symmeteric

    Args:
      X(ndarray(m,n,)): Data X, m datasets, n features
    
    Returns:
      X_normalized(ndarray(m,n,)): X with z-score normalization
      mu(ndarray(n,)): Mean of n features from X
      sigma(ndarray(n,)): Standard deviation of n features from X
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    X_normalized = (X - mu) / sigma
    return X_normalized, mu, sigma

### Hyper params
alpha = 0.3 # learning rate as by recommended z-score normalization
iters = 100

### Data
path = './data/data.csv'
df = pd.read_csv(path)
df.drop(
    labels=["date", "street", "city", "statezip", "country"],
    axis=1,
    inplace=True
)
data = df.to_numpy(dtype=np.float32)

X = data[:, 1:]
y = data[:, 0]
w = np.zeros((X.shape[1]))
b = 0

### Features need to be normalized: Using z-score normalization
X_normalized, mu, sigma = z_score_normalize(X)

def main(X_normalized, y, w, b, alpha):
    weight_amnt = X_normalized.shape[1]
    cost_data = []
    for iter in range(iters):
        w, b = gradient_descent(X_normalized, y, w, b, alpha)
        cost = jwb(X_normalized, y, w, b)
        cost_data.append(cost)
        if iter % 20 == 0 or iter == iters-1:
            weight_info = ''
            for n in range(len(w)):
                weight_info += f'Weight {n}: {w[n]:.4f}, '
            print(f'Iteration: {iter}, {weight_info} Bias: {b:.4f}, Cost: {cost:.4f}')
    # Plotting data
    fig_grid_size = math.ceil(math.sqrt((weight_amnt+1)))
    fig, ax = plt.subplots(fig_grid_size, fig_grid_size)
    ax[0, 0].set_xlabel('Iters')
    ax[0, 0].set_ylabel('Cost')
    ax[0, 0].plot(cost_data)
    for i in range(weight_amnt):
        n = i+1
        fig_col = (n % fig_grid_size)
        fig_row = math.floor(n / fig_grid_size)
        ax[fig_row, fig_col].set_xlabel(df.columns[i+1])
        ax[fig_row, fig_col].set_ylabel("Price")
        ax[fig_row, fig_col].scatter(X[:, i], y, marker="x", color="red", label="target")
        ax[fig_row, fig_col].scatter(X[:, i], [predict(x, w, b) for x in X_normalized], label="predict")
        ax[fig_row, fig_col].legend()
    if weight_amnt < fig_grid_size**2:
        for n in range(weight_amnt, fig_grid_size**2):
            fig_col = (n % fig_grid_size)
            fig_row = math.floor(n / fig_grid_size)
            ax[fig_row, fig_col].set_visible(False)
    fig.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()
if __name__ == '__main__':
    main(X_normalized, y, w, b, alpha)