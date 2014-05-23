'''
Created on Apr 3, 2013

@author: ejhumphrey
'''


import numpy as np
from deeplearn.base import randint, randitem
import theano
import theano.tensor as T
from scipy.cluster.vq import kmeans
import time

safety = {}

def online_vq(X, k, eta, max_iter=5000, init_mode='points',
              warmup=0, di_updates=0, whiten=False, mode='dot',k_sub=False):
    """
    X : np.ndarray
        data, shaped (samples, d0, d1, ... dn)
    k : int, or np.ndarray where k.shape[0] is the number of elements
        number of dict elements
    eta : scalar or 1d-vector
        update rate. if vector, len(eta)>=max_iter
    max_iter : int
        maximum number of update iterations
    init_mode : str
        one of ['points','random'], for initialization strategy
    warmup : int
        number of iterations to visit each dict element *before*
        actually running the algorithm
    di_updates : int, default=0
        number of disjoint information updates per iteration
    whiten : bool, default=False
        standardize (center) the data
    mode : str, default='dot'
        one of ['dot','l2']
    
    """
    
    res = {'D':None,
           'dist':None,
           'mu':None,
           'sd':None,
           'D_init':None}
    time_key = time.asctime()
    
    safety[time_key] = res
    if whiten:
        mu, sd = X.mean(axis=0), X.std(axis=0)
        res['mu'] = mu[np.newaxis,:]
        res['sd'] = sd[np.newaxis,:]
        X = (X-mu)/sd
        
    INIT_MODES = ['points','random','predef']
    assert init_mode in INIT_MODES
    N = X.shape[0]
    D = X.shape[1]
    
    # Initialize Dictionary, if necessary
    if isinstance(k, np.ndarray):
        init_mode = 'predef'
        W = k.copy()
        k = W.shape[0]    
    elif init_mode==INIT_MODES[0]:
        # points in X
        W = np.array([X[n] for n in np.random.permutation(N)[:k]])
        print W.shape
    elif init_mode==INIT_MODES[1]:
        # draw random points with same mean and quarter-sd as the data
        W_shape = [k] + list(X.shape[1:])
        mu, sd = X.mean(axis=0), X.std(axis=0)
        W = np.random.normal(size=W_shape)
        W = (W + mu.reshape(1,D))*sd.reshape(1,D)/2.0
    
    # Handle scalar or vector learning rates
    eta = np.asarray(eta)
    if eta.ndim==0:
        eta = np.zeros(max_iter) + eta
    else:
        max_iter = len(eta)
        
    # Warmup iterations
    for _n in range(warmup):
        for _k in range(k):
            xi = randitem(X)
            # Update W[k] <- W[k] + (xi - W[k])*eta 
            dW = xi - W[_k]
            W[_k] += dW*eta[0]
    
    DONE = False
    count = 0
    loss = []
    res['dist'] = loss
    res['D_init'] = W.copy()
    print "Starting loop..."
    taves = [time.time()]
    while not DONE:
        try:
            # Pick a datapoint in X
            xi = randitem(X)
                
        # Minimize reconstruction error
            # Find nearest k index
            if mode=='l2':
                # d = np.power(W-xi[np.newaxis,...],2.0).sum(axis=1) # np function
                d = l2_fx(xi, W)                                     # theano function
                k_i = d.argmin()
                l = d.min()**0.5
            
            elif mode=='dot':
                a = np.dot(W,xi[:,np.newaxis]).flatten()
                k_i = a.argmax()
                l = a.max()
            
            # Update W[k] <- W[k] + (xi - W[k])*eta 
            dW = xi - W[k_i]
            W[k_i] += dW*eta[count]
            
            # Save error (sqrt here, everything else is a waste of comp.)
            loss += [l]
                
        # Disjoint Penalty
            for _i in range(di_updates):
                # Compute Lookahead
                if k_sub:
                    idx = np.random.permutation(k)[:k_sub]
                else:
                    idx = np.arange(k)
                dW = xi[np.newaxis,:] - W[idx]
                Wp = W[idx] + dW*eta[count]
                if mode=='dot':
                    a_max = disjoint_info_dot(W[idx], p_norm=False)
                    a_max_p = disjoint_info_dot(W[idx], Wp, p_norm=False)
                    i = (a_max_p - a_max).argmin()
                    k_i = idx[i]
                else:
                    raise ValueError("the fuck are you thinking")
                
                W[k_i] = Wp[i]
        
            count += 1
            if count >= max_iter:
                DONE = True
            else:
                # early stopping
                pass
            if (count % 100) == 0:
                taves += [time.time()]
                t_elapsed = np.sum(np.abs(np.diff(taves)))
                t_remain = (max_iter - count) * (t_elapsed/count)
                print "Iter %d : %0.3f sec remain"%(count, t_remain)
        except KeyboardInterrupt:
            break        
        
    
    res['D'] = W
    res['dist'] = np.asarray(loss).squeeze()
    
    no_change = (np.abs(W - res['D_init']).sum(axis=1)==0)
    if no_change.sum():
        print "Warning: %d of %d items did not change"%(no_change.sum(), k)
    res['change'] = np.invert(no_change)
    
    return res


def online_vq2(dset, k, eta, max_iter=5000, init_mode='points',
              warmup=0, di_updates=0):
    """
    dset : Dataset
        data object, with dset.random_sample()
    k : int
        number of dict elements
    eta : scalar or 1d-vector
        update rate. if vector, len(eta)>=max_iter
    max_iter : int
        maximum number of update iterations
    init_mode : str
        one of ['points','random'], for initialization strategy
    warmup : int
        number of iterations to visit each dict element *before*
        actually running the algorithm
    di_updates : int, default=0
        number of disjoint information updates per iteration
    
    """
    
    res = {'D':None,
           'dist':None,
           'mu':None,
           'sd':None,
           'D_init':None}
    
    INIT_MODES = ['points','random']
    assert init_mode in INIT_MODES
    x_i = dset.random_sample(1).squeeze()
    D = len(x_i)
    W_shape = [k,D]
    
    # Initialize Dictionary
    if init_mode==INIT_MODES[0]:
        # points in X
        W = np.array([dset.random_sample().squeeze() for _n in range(k)])

    elif init_mode==INIT_MODES[1]:
        # draw random points with same mean and quarter-sd
        W = np.random.normal(size=W_shape)
        W /= 4.0
    
    # Handle scalar or vector learning rates
    eta = np.asarray(eta)
    if eta.ndim==0:
        eta = np.zeros(max_iter) + eta
    else:
        max_iter = len(eta)
        
    # Warmup iterations
    for _n in range(warmup):
        for _k in range(k):
            xi = dset.random_sample(1).squeeze()
            # Update W[k] <- W[k] + (xi - W[k])*eta 
            dW = xi - W[_k]
            W[_k] += dW*eta[0]
    
    DONE = False
    count = 0
    loss = []
    res['D_init'] = W.copy()
    while not DONE:
        # Pick a datapoint in X
        xi = dset.random_sample(1).squeeze()
            
        # Minimize reconstruction error
        
        # Find nearest k index
        # d = np.power(W-xi[np.newaxis,...],2.0).sum(axis=1) # np function
        d = l2_fx(xi, W)                                     # theano function
        k_min = d.argmin()
        
        # Update W[k] <- W[k] + (xi - W[k])*eta 
        dW = xi - W[k_min]
        W[k_min] += dW*eta[count]
        
        # Save error (sqrt here, everything else is a waste of comp.)
        loss += [d.min()**0.5]
            
        # Disjoint Penalty
        for _i in range(di_updates):
            # Compute Lookahead  
            dW = xi[np.newaxis,:] - W
            Wp = W + dW*eta[count]
            # d_min = disjoint_info(W, Wp)  # np function
            d_min = di_fx(W, Wp)            # theano function, faster
            k_max = d_min.argmax()
            W[k_max] = Wp[k_max]
    
        count += 1
        if count >= max_iter:
            DONE = True
        else:
            # early stopping
            pass
    
    res['D'] = W
    res['dist'] = np.asarray(loss).squeeze()
    
    no_change = (np.abs(W - res['D_init']).sum(axis=1)==0)
    if no_change.sum():
        print "Warning: %d of %d items did not change"%(no_change.sum(), k)
    res['change'] = np.invert(no_change)
    return res

def disjoint_info(D,Dp=None):
    K,R = D.shape 
    dist_mat = np.sqrt(np.power(D[:,np.newaxis,:] - Dp[np.newaxis,:,:],2.0).sum(axis=-1))
    dist_mat[np.eye(N=K, dtype=bool)] = np.inf
    d_min = dist_mat.min(axis=0)
    return d_min

def disjoint_info_theano(D,Dp=None):
    """
    D : theano matrix
    Higher is better, take negative to minimize, or subtract
    L2 norm
    """
    
    if Dp is None:
        Dp = D
    
    K = D.shape[0]
    R = D.shape[1] 
    
    # Compute the distance matrix
    dist_mat = T.sqrt(T.pow(D.dimshuffle(0,'x',1) - Dp.dimshuffle('x',0,1),2.0).sum(axis=-1))
    
    # Create a diagonal matrix where the elements are greater than anything in dist_mat
    max_mask = T.identity_like(dist_mat)*(T.max(dist_mat) + 10.0)
    
    # Find the min distance along each column
    d_min = T.min(dist_mat + max_mask, axis=0)
    
    # return the min-distances
    return d_min

def disjoint_info_dot(D,Dp=None,p_norm=False):
    """
    D : theano matrix
    Higher is better, take negative to minimize, or subtract
    L2 norm
    """
    
    if Dp is None:
        Dp = D
    
    if p_norm:
        D = lp_norm(D,p_norm)
        Dp = lp_norm(Dp,p_norm)
        
    
    K = D.shape[0]
    R = D.shape[1] 
    
    # Compute the activation matrix
    act_mat = np.array([np.dot(D,di[:,np.newaxis]).flatten() for di in Dp])
    
    # Create a diagonal matrix where the elements are greater than anything in dist_mat
    act_mat[np.eye(N=K, dtype=bool)] = 0.0
    a_max = act_mat.max(axis=0)
    return a_max
    
def gen_di_fx():
    d1 = T.matrix('W', dtype='float32')
    d2 = T.matrix('Wp', dtype='float32')
    return theano.function(inputs=[d1,d2],
                           outputs=disjoint_info_theano(d1, d2),
                           allow_input_downcast=True)

def gen_l2_fx():
    x = T.vector('X', dtype='float32')
    W = T.matrix('W', dtype='float32')
    l2 = T.pow(W - x.dimshuffle('x',0),2.0).sum(axis=-1)
    return theano.function(inputs=[x,W],
                           outputs=l2,
                           allow_input_downcast=True)

def lp_norm(x,p=2.0):
    """
    x : np.ndarray with shape (N,R)
        N are IID samples and R is the coordinate representation 
        
    """
    flatten = False
    if x.ndim==1:
        x = x[np.newaxis,:]
        flatten = True
    u = np.power(np.power(np.abs(x),p).sum(axis=-1), 1.0/p)
    u[u==0] = 1.0
    y = x / u[:,np.newaxis]
    if flatten:
        y = y.flatten()
    return y

#di_fx = gen_di_fx()
#l2_fx = gen_l2_fx()