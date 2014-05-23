'''
Created on Oct 25, 2012

@author: ejhumphrey

Theano operations - entirely procedural
'''

import theano
import theano.tensor as T
import numpy as np

from scipy.signal.windows import gaussian
from deeplearn.core import FLOATX


def cross_entropy(x,y):
    """
    pretty sure x and y need to be bounded on (0,1)
    """
    return -T.mean(x * T.log(y) + (1.0 - x) * T.log(1.0 - y), axis=1 )    

def l2(x, y):
    return T.sqrt(T.sum(T.pow(x - y, 2.0), axis=1))

def l1(x, y):
    return T.sum(T.abs_(x - y), axis=1)

def tanimoto(x,y):
    num = T.sum(x*y, axis=1)
    den0 = T.sum(x*x, axis=1)
    den1 = T.sum(y*y, axis=1)
    return 1.0 - num / (den0+den1-num) 

def shrinkage(x, beta, theta):
    """
    """
    return T.sgn(x)*((1.0/beta) * (T.log(T.exp(beta*theta) + T.exp(beta * T.abs_(x)) - 1.0)) - theta)

def np_shrinkage(x, beta, theta):
    """
    """
    return (1.0/beta) * (np.log(np.exp(beta*theta) + np.exp(beta * np.abs(x)) - 1.0)) - theta

def tanh_abs_shrinkage(x):
    return T.tanh(shrinkage(x,5.0,0.5))

def tanh_shrinkage(x):
    return T.tanh(T.sgn(x)*shrinkage(x,5.0,0.5))

def sigmoid(x):
    """
    """    
    return theano.tensor.nnet.sigm.sigmoid(x)

def noop(x):
    return x

def linear(x,m=1.0):
    return m*x

def sine(x,a=1.0):
    return T.sin(a*x)

def sinexp(x,p=2.0):
    return T.pow(T.sin(x),p)
    

def soft_hinge(x, m=0.0, Q=1.0):
    """
    x : symbolic, or scalar
        typically, the independent variable
    m : scalar, or symbolic
        typically, the margin or offset
    Q : scalar
        knee of the log-approx
    
    note: standard behavior is monotonically increasing; swapping a and b 
        will flip the function horizontally.
    """
    return T.log(1 + T.exp(Q * (x - m))) / Q

def soft_abs(x, Q=100.0):
    """
    due to the log-approx, soft_abs(0) >= 0.0
    """
    # TODO: subtract 2*soft_hinge(0)
    return soft_hinge(x=x, m=0.0, Q=Q) + soft_hinge(x=0.0, m=x, Q=Q)

def npshingle(x,N,h,FLAT=True):
    """
    valid shingle op
    """
    
    M = x.shape[0] - N + 1
    X = np.zeros([M, N] + list(x.shape[1:]),dtype=x.dtype)
    for m in range(M):
        idx = int(np.round(h*m))
        x_m = x[idx:idx+N]
        X[m,:x_m.shape[0]] = x_m
    
    if FLAT:
        return npflatmat(X)
    else:
        return X
    
def shrinkage2(x, Q, margin):
    pos = T.log(1 + T.exp(Q * (x - margin))) / Q
    neg = T.log(1 + T.exp(Q * (-x + margin))) / Q
    return T.tanh(pos+neg)

def shrinkage3(x, beta, b=0.5, G=1.5):
    x = T.tanh(x)*G
    return T.sgn(x)*((1.0/beta) * (T.log(T.exp(beta*b) + T.exp(beta * T.abs_(x)) - 1.0)) - b)

def npflatmat(X):
    """
    reduce a tensor to a matrix, such that the first
    dimension is preserved and all others are collapsed.
    """
    assert X.ndim>=2
    shp = X.shape
    return X.reshape(shp[0], np.prod(shp[1:]))

def npunshingle(Z,N):
    assert Z.ndim>=2
    shp = Z.shape
    return Z.reshape(shp[0]*N, np.prod(shp[1:])/N)

def shingle(X,N,h,M):
    """
    X : matrix
        duh
    N : int
        frames to concat
    h : int
        hopsize
    M : int
        num points out
    """
    return T.concatenate([X[T.arange(m, m+N)].flatten(1).dimshuffle('x',0) \
                          for m in np.arange(M,step=h)])
    
#def unshingle(X,N):
#    nshp = T.stack(N, T.shape(X)[1]/N)
#    return T.concatenate([X[m].reshape((),ndim=2) for m in np.arange(M)])
#    
    

def make_gaussian(gdim):
    G = [gaussian(gd,gd/4.,True) for gd in gdim]
    G = G[0][:,np.newaxis]*G[1][np.newaxis,:]    
    G /= G.sum()
    return G

def lcn(x_in,g_shared):
    """
    x_in : tensor4
        shape (N, 1, d0, d1) 
    
    g_shared : theano.shared
        shape (1, 1, d0, d1)
    """
    gshp = g_shared.get_value().shape
    Xh = T.nnet.conv.conv2d(input=x_in,
                            filters=g_shared,
                            filter_shape=gshp,
                            border_mode='full')
    d0 = gshp[2]/2
    d1 = gshp[3]/2
    N = x_in.shape[2]
    M = x_in.shape[3]
    
    V = x_in - Xh[:,:,d0:d0+N, d1:d1+M]
    Vh = T.nnet.conv.conv2d(input=T.pow(V,2.0),
                            filters=g_shared,
                            filter_shape=gshp,
                            border_mode='full')
    S = T.sqrt(Vh[:,:,d0:d0+N, d1:d1+M])
    S_mu = T.mean(T.mean(S,axis=-1),axis=-1).dimshuffle(0,1,'x','x')
    mask_lt = T.cast(T.lt(S, S_mu),FLOATX)
    mask_ge = T.cast(T.ge(S, S_mu),FLOATX)
    S2 = mask_lt*S_mu + mask_ge*S
    return V / S2
 
    
    
    