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
import numpy.linalg as linalg


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
    L = linalg.cholesky(matrix)
    return 2*np.sum(np.log(np.diag(L)))