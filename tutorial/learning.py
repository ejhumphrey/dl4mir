'''
Created on Oct 29, 2013

@author: ejhumphrey
'''
from sklearn.datasets.samples_generator import make_regression, make_blobs
import numpy as np
from matplotlib.pyplot import figure, show

def line_example(b_range=5, m_range=50, num_steps=101):

    X, Y = make_data(200, 1.3, noise=.3)
    loss_surface = np.zeros([num_steps, num_steps])
    min_val = np.inf
    min_mb = None
    m_range = np.linspace(-m_range, m_range, num_steps)
    b_range = np.linspace(-b_range, b_range, num_steps)
    for i, m in enumerate(m_range):
        for j, b in enumerate(b_range):
            loss_surface[i, j] = mse(X, Y, m, b)
            if loss_surface[i, j] < min_val:
                min_val = loss_surface[i, j]
                min_mb = (m, b)

    fig = figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(X.flatten(), Y.flatten())
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Observed Data")

    x_opt = np.linspace(X.min(), X.max(), 5)
    y_opt = x_opt * min_mb[0] + min_mb[1]
    ax1.plot(x_opt, y_opt)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(np.flipud(loss_surface.T), interpolation='nearest', aspect='equal')
    ax2.set_ylabel("Y-Intercept (b)")
    ax2.set_xlabel("Slope (m)")
    ax2.set_title("Loss Surface (MSE)")
    ax2.set_xticks(np.arange(num_steps)[:: num_steps / 10])
    ax2.set_xticklabels(["%0.2f" % v for v in m_range[::num_steps / 10]])
    ax2.set_yticks(np.arange(num_steps)[:: num_steps / 10])
    ax2.set_yticklabels(["%0.2f" % v for v in b_range[::-num_steps / 10]])
    show()


def mse(x, y_true, m, b):
    y_pred = x * m + b
    return np.sqrt(np.power(y_true - y_pred, 2.0)).mean()


def make_data(num_points, theta, x_lim=5, noise=1.0):
    x = np.random.uniform(low= -x_lim, high=x_lim, size=(num_points,))
#    m = np.random.uniform(low= -m_lim, high=m_lim)
    n = np.random.normal(scale=noise, size=(num_points,))
    return x, x + np.sin(x * theta) + n
