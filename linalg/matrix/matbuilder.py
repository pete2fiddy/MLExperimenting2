import numpy as np


def permutation_mat(rows, order):
    P = np.zeros((rows, rows), dtype = np.float64)
    for i in range(0, P.shape[0]):
        P[i, order[i]] = 1
    return P

def row_swap_mat(rows, swap_rows):
    swap = np.identity(rows)
    swap[[swap_rows[0], swap_rows[1]]] = swap[[swap_rows[1], swap_rows[0]]]
    return swap


'''creates a matrix E that, when:
EA,
yields a matrix where:
A[row r1] <- A[row r1] + n*A[row r2]
'''
def row_add_mat(rows, r1r2, n):
    mat = np.identity(rows)
    mat[r1r2[0],r1r2[1]] = n
    return mat


def rand_mat(dim, range):
    return range[0] + (np.random.rand(dim) * (range[1]-range[0]))
