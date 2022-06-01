# -*- coding: utf-8 -*-
"""
ggspy.matrix
-------------

Methods for Matrix manipulation.

"""
# Imports

# built-in

# local

# 3rd-party
import numpy as np

# CardioID


def cholesky(matrix):
    """
    Cholesky Factorization.

    Parameters
    ----------
    matrix : n-by-n array
        Matrix to decompose
    
    Returns
    -------
    L : n-by-n array
        Decomposition factor L.
    """

    n = matrix.shape[0]
    L = np.zeros(matrix.shape)
    for j in range(n):
        sum_ = np.sum(L[j, :j]**2)
        L[j, j] = np.sqrt(matrix[j, j] - sum_)

        for i in range(j, n):
            sum_ = np.sum(L[i, :j]*L[j, :j])
            L[i][j] = (matrix[i, j] - sum_)/L[j, j]
    

    return L


def log_det(matrix):
    """
    Compute the log-determinant of a
    positive definite matrix using its
    Cholensky decomposition.

    Parameters
    ----------
    matrix : n-by-n array
        matrix.

    Returns
    -------
    float
        Log-det of the matrix.
    """
    L = cholesky(matrix)
    return 2*np.sum(np.log(np.diag(L)))


def trace_inv(matrix):
    np.trace(np.linalg.inv(matrix))