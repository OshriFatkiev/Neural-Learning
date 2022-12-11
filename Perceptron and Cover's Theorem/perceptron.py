import numpy as np


def perceptron(X, y0, lr=0.1, t_max=10000):
    """
        The Perceptron learning algorithm

        Parameters
        ----------
        X : NxP REAL MATRIX, where P is the training set size
            and N is the numbers of features.
        y0 : Px1 REAL VECTOR
            All training labels, y_i = label(x_i)
        lr: FLOAT, optional
            Learning rate. The default is .1
        t_max : INT, optional
            max runtime in epochs. The default is 1e4.

        Returns
        -------
        w : 1XN REAL VECTOR
            The weight to give each feature in the prediction of new samples.
        converged : BOOLEAN
            whether the algorithm converged (no more updates) or reached t_max and has stopped.
        epochs : INT
            The number of epochs.

    """
    assert X.shape[1] == y0.shape[0], "input-output set size mismatch"
    y0 = y0.reshape(-1, 1)
    w = np.random.random(X.shape[0]).reshape(-1, 1)
    epochs = 0
    converged = False

    while (not converged) and (epochs < t_max):
        converged = True
        res = (w.T @ X * y0.T).reshape(-1, 1)
        violation_idxs = np.where(res < 0)
        if np.count_nonzero(violation_idxs) != 0:
            for i in violation_idxs:
                w += lr * X[:, i] @ y0[i]
            converged = False
        epochs += 1

    return w, converged, epochs



