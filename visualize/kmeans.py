'''
Created on Oct 2, 2013

@author: ejhumphrey
'''


from matplotlib.pyplot import show, figure, Circle
import numpy as np
from sklearn import datasets
from sklearn.decomposition.pca import PCA

def soft_thresh(x, theta):
    return np.maximum(np.zeros(x.shape), x - theta)

iris = datasets.load_iris()
data = iris.data[:, :2]

fig = figure()
ax = fig.gca()
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
unit_circle = Circle((0, 0), 1,
                     color='k',
                     linestyle='dashdot',
                     alpha=0.5, fill=False)
ax.add_artist(unit_circle)

num_samples = 12
num_clusters = 10
rand_idx = np.random.permutation(len(data))[:num_samples]
mu = np.mean(data, axis=0)[np.newaxis, :]
var = np.var(data, axis=0)[np.newaxis, :]
eta = var * 0.02

data = ((data - mu) / np.power(var + eta, 0.5))
x, y = data[rand_idx].T
#ax.scatter(x, y, s=25, c='b', marker='o', linewidths=1.5)

pca = PCA(n_components=2, whiten=True)

data_whitened = pca.fit_transform(data)
x, y = data_whitened[rand_idx].T
ax.scatter(x, y, s=25, c='g', marker='o', linewidths=1.5)
[ax.annotate("%d" % i, (x[i], y[i])) for i in range(num_samples)]

D = np.random.normal(0, 1, size=(num_clusters, 2))
Dsphere = (D / np.power(np.power(D, 2.0).sum(axis=1), 0.5)[:, np.newaxis])

Dx, Dy = Dsphere.T
ax.scatter(Dx, Dy, s=25, c='r', marker='x', linewidths=1.5)
[ax.annotate("%d" % i, (Dx[i], Dy[i])) for i in range(num_clusters)]

A = np.dot(data_whitened[rand_idx], Dsphere.T)
fig = figure()
ax = fig.add_subplot(311)
ax.imshow(A, interpolation='nearest', aspect='auto')
ax = fig.add_subplot(312)
ax.imshow(soft_thresh(A, 0.2), interpolation='nearest', aspect='auto')
ax = fig.add_subplot(313)
ax.imshow(soft_thresh(A, 0.5), interpolation='nearest', aspect='auto')
show()
