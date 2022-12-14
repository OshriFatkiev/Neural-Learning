# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy


def adatron(K, y0, eps=0.0001, t_max=1000):
    """
    The AdaTron algorithm for SVM learning
    
    Parameters
    ----------
    K : PxP REAL MATRIX, where P is the training set size
        Contains all pairwise overlaps between training examples: X_ij= x_i*x_j
    y0 : Px1 REAL VECTOR
        All training labels, y_i = label(x_i)
    eps: FLOAT, optional
        Stopping criterion (when update<eps). The default is .0001
    t_max : INT, optional
        max runtime in epochs. The default is 1000.

    Returns
    -------
    converged : BOOLEAN
        whether the algorithm converged (no more updates) or reached t_max and eas stopped.
    A : ExP REAL MATRIX, where E is the number of epochs
        Contains the P-dim support coefficient vectors (alpha) for all the epochs.
        A[t,:] is the support vector alpha_t (at epoch=t)
        When the algorithm has converges, A[-1,:] is the final result that defines the decision rule 

    """
    P, P1 = K.shape
    assert P == P1, "Kernel matrix K should be PxP, where P is the training set size"
    assert y0.size == P, "input-output set size mismatch"
    converged = False
    epochs = 0
    A = []
    eta = 0.2/(np.max(np.linalg.norm(K, axis=0)))
    A.append(np.random.rand(P, 1))
    
    while (not converged) and (epochs < t_max):
        update_flag = False
        a = deepcopy(A[-1])
        for mu in range(P):
            y = y0[mu]
            coeff = y*(a*y0).T
            da = max(-a[mu], eta*(1-coeff@K[:, mu]))
            if np.isinf(da):
                print("stopping because of exploding updates, epoch={}, mu={}".format(epochs, mu))
                converged = True
                continue;
            if np.isnan(da):
                print("nan")
                converged = True
                continue;
            update_flag = update_flag or (da > eps)
            a[mu] += da
        converged = converged or (not update_flag)
        epochs += 1
        A.append(a)
        
    return converged, np.squeeze(np.array(A))
