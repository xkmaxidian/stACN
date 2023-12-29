import numpy as np
from numpy.linalg import svd
from scipy.spatial.distance import cdist


def soft_numpy(x, T):
    """
   Apply the soft thresholding operator to the input array using NumPy.

   Soft thresholding is a component of various sparse coding algorithms. It shrinks the input values towards zero, potentially setting some to zero if they are below the threshold T, which can be useful for denoising and regularization.

   Parameters:
   - x: A NumPy array containing the input data to be thresholded.
   - T: A non-negative threshold value or array. If T is an array, it should be broadcastable to the shape of x.

   Returns:
   - y: A NumPy array containing the result of applying the soft thresholding operator to the input data. Each element in x is reduced by the threshold T if its absolute value is greater than T; otherwise, it is set to zero.
   """
    if np.sum(np.abs(T)) == 0.:
        y = x
    else:
        y = np.maximum(np.abs(x) - T, 0.)
        y = np.sign(x) * y
    return y


def soft(x, T):
    """
   Apply the soft thresholding operator to the input array.

   This operation is commonly used in signal processing and statistics, particularly within the context of lasso regression and other methods that involve regularization. It effectively shrinks the input values towards zero by the threshold T.

   Parameters:
   - x: An array-like object containing the input data to be thresholded.
   - T: A non-negative threshold value. If T is an array, it should be of the same shape as x.

   Returns:
   - y: The result of applying the soft thresholding operator to the input data. Values are reduced by the threshold amount T, and values with absolute value less than T are set to zero.
   """
    if np.sum(np.abs(T)) == 0.:
        y = x
    else:
        y = np.maximum(np.abs(x) - T, 0.)
        y = np.sign(x) * y
    return y


def create_sppmi_mtx(G, k):
    """
    Create a Shifted Positive Pointwise Mutual Information (SPPMI) matrix from a given co-occurrence matrix.

    Parameters:
    - G: A co-occurrence matrix where each element G[i, j] represents the co-occurrence count between items i and j.
    - k: A shifting parameter used to reduce the effect of large co-occurrence counts.

    Returns:
    - sppmi: A matrix of the same shape as G, where each element contains the SPPMI score between items i and j.
    """
    node_degrees = np.array(G.sum(axis=0)).flatten()
    node_degrees2 = np.array(G.sum(axis=1)).flatten()
    W = np.sum(node_degrees)

    sppmi = np.zeros_like(G).astype(float)
    row, col = np.nonzero(G > 0)

    for index in range(len(col)):
        i = row[index]
        j = col[index]
        score = np.log(G[i][j] * W / (node_degrees2[row[index]] * node_degrees[col[index]])) - np.log(k)
        sppmi[row[index], col[index]] = max(score, 0.0)

    return sppmi


def solve_l1l2(W, lamb):
    """
    Solve the L1-L2 optimization problem for each row of the input matrix W.

    This function applies L2 regularization (also known as ridge regression) to each row of the matrix W, with a regularization parameter lambda. It is often used in the context of regression problems where both L1 and L2 regularization are applied, hence the name 'l1l2'.

    Parameters:
    - W: A 2D NumPy array where each row represents a set of variables to be optimized.
    - lamb: The regularization parameter (lambda) that controls the strength of the shrinkage applied to the variables.

    Returns:
    - E: A 2D NumPy array of the same shape as W, where each row has been adjusted according to the L2 regularization.
    """
    n = W.shape[0]
    E = W.copy()

    for i in range(n):
        E[i, :] = solve_l2(W[i, :], lamb)
        # print(E[:, i])
    return E


def solve_l2(w, lamb):
    """
    Apply L2 regularization to a vector w with a regularization parameter lambda.

    This function shrinks the input vector w towards zero by the regularization parameter lambda, but only if the norm of w is greater than lambda. If the norm of w is less than or equal to lambda, the function returns a vector of zeros. This is a typical operation in ridge regression and related regularization techniques.

    Parameters:
    - w: A 1D NumPy array representing the vector to be regularized.
    - lamb: The regularization parameter (lambda) that controls the strength of the shrinkage. If the norm of w is greater than lambda, w is shrunk towards zero; otherwise, it is set to zero.

    Returns:
    - x: A 1D NumPy array representing the regularized vector. If the norm of w is greater than lambda, the returned vector is w shrunk towards zero; otherwise, it is a vector of zeros.
    """
    nw = np.linalg.norm(w)
    # print(nw)
    if nw > lamb:
        # print(w)
        x = (nw - lamb) * w / nw
    else:
        x = np.zeros_like(w)
    return x


def opt_p(Y, mu, A, X):
    """
    Perform an optimization operation on the input parameters Y, mu, A, and X using Singular Value Decomposition (SVD).

    Parameters:
    - Y: Input matrix.
    - mu: Scalar value.
    - A: Input matrix.
    - X: Input matrix, each column is a data point.

    Returns:
    - P: Resultant matrix after performing the optimization operation.
    """
    G = X.T
    Q = (A - Y / mu).T
    # Q = (Y / mu-A ).T
    W = np.dot(G.T, Q) + np.finfo(float).eps
    U, S, Vt = svd(W, full_matrices=False)
    # U, S, Vt = svd(W, 0)
    PT = np.dot(U, Vt)
    P = PT.T
    return P


def construct_w_pkn(X, k=5, issymmetric=1):
    """
    Construct similarity matrix W using the PKN algorithm.

    Parameters:
    - X: Each column is a data point.
    - k: Number of neighbors.
    - issymmetric: Set W = (W + W')/2 if issymmetric=1.

    Returns:
    - W: Similarity matrix.
    """
    dim, n = X.shape
    D = cdist(X.T, X.T, metric='euclidean') ** 2

    idx = np.argsort(D, axis=1)  # sort each row

    W = np.zeros((n, n))
    for i in range(n):
        id = idx[i, 1:k + 2]
        di = D[i, id]
        W[i, id] = (di[k] - di) / (k * di[k] - np.sum(di[:k]) + np.finfo(float).eps)

    if issymmetric == 1:
        W = (W + W.T) / 2

    return W


def wshrink_obj(x, rho, sX, isWeight, mode):
    """
   This function performs weighted shrinkage on the input data 'x' using Singular Value Decomposition (SVD)
   and Fast Fourier Transform (FFT). The shrinkage factor is controlled by 'rho'. The shape of the input data
   is given by 'sX'. If 'isWeight' is 1, a weight matrix is calculated. The 'mode' parameter determines how
   the axes of the input data are swapped or moved for the FFT operation.
   """
    if isWeight == 1:
        C = np.sqrt(sX[2] * sX[1])
    if mode is None:
        mode = 1
    X = x.reshape(sX)
    if mode == 1:
        Y = np.swapaxes(X, 0, 2)
    elif mode == 3:
        Y = np.moveaxis(X, 0, -1)
    else:
        Y = X

    Yhat = np.fft.fft(Y, axis=2)
    objV = 0

    if mode == 1:
        n3 = sX[1]
    elif mode == 3:
        n3 = sX[0]
    else:
        n3 = sX[2]

    endValue = np.int16(np.floor(n3 / 2) + 1)

    for i in range(endValue):
        uhat, shat, vhat = svd((Yhat[:, :, i]), full_matrices=False)
        if isWeight:
            weight = C / (np.diag(shat) + np.finfo(float).eps)
            tau = rho * weight
            shat = soft(shat, np.diag(tau))
        else:
            tau = rho
            shat = np.maximum(shat - tau, 0)
        objV += np.sum(shat)
        Yhat[:, :, i] = np.dot(np.dot(uhat, np.diag(shat)), vhat)
        if i > 1:
            Yhat[:, :, n3 - i] = np.dot(np.dot(np.conj(uhat), np.diag(shat)), np.conj(vhat))
            objV += np.sum(shat)

    Y = np.fft.ifft(Yhat, axis=2)
    Y = np.real(Y)

    if mode == 1:
        X = np.fft.ifft(Y, axis=2)
    elif mode == 3:
        X = np.moveaxis(Y, -1, 0)
    else:
        X = Y

    x = X.flatten()
    return x, objV
