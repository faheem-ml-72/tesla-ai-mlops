from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np

def train_gp_model(data):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data

    kernel = C(1.0) * RBF(length_scale=10)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

    gp.fit(X, y)
    return gp