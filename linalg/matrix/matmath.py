import numpy as np
import linalg.matrix.matbuilder as matbuilder

NEAR_ZERO_TOLERANCE = 0.0000000000000001

def PLU_decompose(A):
    #Factors M in the factorization A = MU so that M = PL, then returns P, L, U in the factorization, A = PLU
    #The diagonal of L is 1
    L, U = upper_triangular_factor(A)
    np.testing.assert_array_almost_equal(np.dot(L, U), A)
    row_order = np.zeros(L.shape[0], dtype = np.int)
    for i in range(L.shape[0]):
        num_zeros_in_row = 0
        for j in range(L.shape[1]-1, -1, -1):
            if L[i,j] == 0:
                num_zeros_in_row += 1
            else:
                row_order[i] = (L.shape[1] - 1) - num_zeros_in_row
                break
    P = matbuilder.permutation_mat(L.shape[0], row_order)
    return P, np.dot(P.T, L), U

def inv(A):
    #determines the inverse of A using the inverses of A's factors, P, L, and U, whose constraints make them easy to invert
    #its advantage over simple gauss-jordan is that the ease of inversion of P, L, and U yields better-order run-time complexity
    #inv(A) = inv(PLU) = inv(U)inv(L)inv(P)
    assert A.shape[0] == A.shape[1], "matrix to be inverted is not square"
    P, L, U = PLU_decompose(A)
    P_inv = P.T
    L_inv = __inv_lower_triangular_mat(L)
    U_inv = __inv_upper_triangular_mat(U)
    return multimul([U_inv, L_inv, P_inv])

def __inv_lower_triangular_mat(L):
    assert L.shape[0] == L.shape[1], "lower triangular matrix is not square, cannot invert"
    L = L.copy()
    aug = np.identity(L.shape[0])
    for pvt_col in range(0, min(L.shape[0], L.shape[1])):
        for i in range(pvt_col+1, L.shape[0]):
            add_amount = -L[i, pvt_col]/L[pvt_col,pvt_col]
            aug[i] += add_amount*aug[pvt_col]
            L[i] += add_amount*L[pvt_col]
    return aug

def __inv_upper_triangular_mat(U):
    assert U.shape[0] == U.shape[1], "upper triangular matrix is not square, cannot invert"
    U = U.copy()
    aug = np.identity(U.shape[1])
    for pvt_col in range(U.shape[1]-1, -1, -1):
        for i in range(pvt_col-1, -1, -1):
            add_amount = -U[i, pvt_col]/U[pvt_col, pvt_col]
            aug[i] += add_amount*aug[pvt_col]
            U[i] += add_amount*U[pvt_col]
        pvt_scale_factor = 1.0/U[pvt_col,pvt_col]
        U[pvt_col] *= pvt_scale_factor
        aug[pvt_col] *= pvt_scale_factor
    return aug




'''
factors A into the product between M and U, where U is an upper-triangular matrix.
Returns:
M, U
'''
def upper_triangular_factor(A):
    U = A.copy()
    M = np.identity(A.shape[0], dtype = np.float64)
    for pivot_col in range(0, U.shape[1]):
        largest_pivot_row = pivot_col
        for i in range(pivot_col+1, A.shape[0]):
            if abs(U[i, pivot_col]) > abs(U[largest_pivot_row, pivot_col]): largest_pivot_row = i
        #is equivalent to M = dot(M, swap_mat^-1)
        #M = dot(M, swap_mat^-1) = dot((swap_mat^-1).T, M.T).T = dot(swap_mat, M.T).T = swap [pivot_col, largest_pivot_row] COLUMNS of M
        M[:, [pivot_col, largest_pivot_row]] = M[:, [largest_pivot_row, pivot_col]]
        U[[pivot_col, largest_pivot_row]] = U[[largest_pivot_row, pivot_col]]
        if abs(U[pivot_col, pivot_col]) < NEAR_ZERO_TOLERANCE: raise ValueError("Singular matrix in upper_triangular_factor")
        for i in range(pivot_col + 1, A.shape[0]):
            if abs(U[i,pivot_col]) >= NEAR_ZERO_TOLERANCE:
                add_amount = -U[i,pivot_col]/U[pivot_col, pivot_col]
                U[i] += add_amount*U[pivot_col]
                #is equivalent to M = dot(M, inv(E))
                #M = dot(M, inv(E)) = dot(inv(E).T, M.T).T = dot(row pivot_col <- row pivot_col - add_amount*row i, M.T).T =
                #dot(col pivot_col <- col pivot_col - add_amount*col i, M)
                M[:,pivot_col] -= add_amount*M[:,i]
            #lossy math makes it so that U[i,pivot_col] isn't always zero, but instead a very very small number. Manually setting to 0 will
            #sometimes make the factorization slightly off, but will prevent non-upper-triangular U
            U[i,pivot_col] = 0
    return M, U



def det(A):
    #Leverages PLU decomposition, treating A = PLU. As such, det(A) = det(P)*det(L)*det(U), where the determinant of permutation and triangular matrices
    #is very easy to find.
    #det(P) = (-1)^S where S is the number of row exchanges in P,
    #det(L and U) = the product of elements along their diagonal. Also, in matmath's A = PLU decomposition, L contains 1s on the diagonal, meaning
    #its determinant is 1
    try:
        P, L, U = PLU_decompose(A)
        det_P = __sign_permutation(P)
        det_L = 1
        det_U = 1
        for i in range(0, min(U.shape[0], U.shape[1])):
            det_U *= U[i,i]
        return det_P*det_L*det_U
    except ValueError:
        #singular matrices have a determinant of 0
        return 0

def column_space(A):
    #returns the vector subspace in which Ax = b is solveable so long that
    #b falls into the subspace (because Ax = x1*A[col 1] + x2*A[col 2] ... )
    #equivalently, returns the independent columns of A. Subspace is in R(#rows in A)
    return None

def nullspace(A):
    #Returns the column basis of the null space of A
    #works with the reduced row echelon form of A,then performs columns swaps to put it in the form of:
    #[[I, F(matrix of free columns)],
    #[0...]]
    #Column swaps have no effect on the solution of Ax = 0, since all they do is rewrite the ordering of the system of equations R represents,
    #so no post-requisites are required to rectify the column swaps.
    #A matrix of this form has a nullspace solution of RN(null space matrix) = 0 ->
    #[[I, F(matrix of free columns)], [[-F],
    #[0...]]                          [I]]
    #by block matrix multiplication, so all that is really needed is F, which can be determined by removing all non-pivot columns of R,
    #and I, which is easily computed
    R, pvts = rref(A)
    F = R[:pvts[pvts.shape[0]-1, 0]+1, :]
    F = np.delete(F, pvts[:,1], axis = 1)
    N = np.zeros((F.shape[0]+F.shape[1], F.shape[1]))
    N[:F.shape[0], :] = -F
    N[F.shape[0]:, :] = np.identity(F.shape[1])
    np.testing.assert_array_almost_equal(np.dot(R, N), np.zeros((R.shape[0], N.shape[1])))
    np.testing.assert_array_almost_equal(np.dot(A, N), np.zeros((A.shape[0], N.shape[1])))
    null_space = [N[:,i] for i in range(N.shape[1])]
    return null_space

def rref(A, print_steps = False, stable_swap_rows = True):
    #returns the reduced row echelon form of A, meaning it is the row echelon form of A with all entries above each pivot in the column
    #of that pivot equalling zero

    #print_steps determines whether or not the steps of the rref are printed
    R, pvts = ref(A, print_steps = print_steps, stable_swap_rows = stable_swap_rows)
    RREF = R.copy()
    steps = ""
    for pvt in reversed(pvts):
        if print_steps: steps += "~\n" + str(RREF) + "\n-->\n"
        if print_steps: steps += "row " + str(pvt[0]) + " /= " + str(RREF[pvt[0],pvt[1]]) + "\n"

        RREF[pvt[0]] /= RREF[pvt[0],pvt[1]]
        if print_steps: steps += "~\n" + str(RREF) + "\n"
        if print_steps: steps += "-->\n"
        for i in range(pvt[0]-1,-1,-1):
            row_multiple = -(RREF[i,pvt[1]]/RREF[pvt[0],pvt[1]])
            if print_steps: steps += "row " + str(i) + " += " + str(row_multiple) + " row " + str(pvt[0]) + "\n"
            RREF[i] += row_multiple*RREF[pvt[0]]
            #may have problems with lossyness, REF[i,pvt[1]] may not be zero at this point, but instead a very small number
    if print_steps: steps += "RREF COMPLETED"
    if print_steps: print(steps)
    return RREF, pvts

def rank(A):
    _, pvts = ref(A)
    return len(pvts)




def ref(A, print_steps = False, stable_swap_rows = True):
    #returns the row echelon form of A, meaning that the first non-zero entry in each row of the result will act like a staircase from top
    #to bottom. Returns:
    #R, pivots(row,col)

    #stable_swap_rows determines whether or not technically uneccessary row swaps are performed in favor of preventing possible overflow,
    #and print_steps determines whether steps to the problem are printed out
    R = A.copy()
    pvt_row = 0
    pvt_col = 0
    pvts = []
    steps = ""
    while pvt_row < R.shape[0] and pvt_col < R.shape[1]:
        if print_steps: steps += "~\n" + str(R) + "\n"
        largest_pvt_row = pvt_row
        if stable_swap_rows:
            for i in range(pvt_row+1, R.shape[0]):
                if abs(R[i,pvt_col]) > abs(R[largest_pvt_row,pvt_col]): largest_pvt_row = i
        #performs a row swap so that the current pvt_row contains the largest element in the pvt_col
        R[[pvt_row, largest_pvt_row]] = R[[largest_pvt_row, pvt_row]]
        if print_steps: steps += "-->\n"
        if abs(R[pvt_row, pvt_col]) >= NEAR_ZERO_TOLERANCE:
            pvts.append(np.array([pvt_row, pvt_col]))
            for i in range(pvt_row+1, R.shape[0]):
                row_multiple = -(R[i,pvt_col]/R[pvt_row,pvt_col])
                if print_steps: steps += "row " + str(i) + " += " + str(row_multiple) + " row " + str(pvt_row) + "\n"
                R[i] += row_multiple*R[pvt_row]
            pvt_row += 1
        pvt_col += 1

    R[np.abs(R) < NEAR_ZERO_TOLERANCE] = 0
    if print_steps: steps += "REF COMPLETED"
    if print_steps: print(steps)
    return R, np.asarray(pvts).astype(np.int)

'''
Precondition: P is a permutation matrix
Calculates the sign of a permutation matrix, P, meaning -1 if the number of row swaps required to permute
is odd, or 1 if the number of row swaps required to permute is even. This is also the determinant of P
'''
def __sign_permutation(P):
    sign = 1
    P = P.copy()
    for i in range(P.shape[0]):
        if P[i,i] != 1:
            for i2 in range(i+1, P.shape[0]):
                if P[i2, i] == 1:
                    P[[i2,i]] = P[[i,i2]]
                    sign *= -1
                    break
    return sign



def is_lower_triangular(A):
    for i in range(0, A.shape[0]):
        for j in range(i+1, A.shape[1]):
            if A[i,j] != 0: return False
    return True

def is_upper_triangular(A):
    for i in range(1, A.shape[0]):
        for j in range(0, i):
            if A[i,j] != 0: return False
    return True


def mul(A, B):
    if A.shape[1] != B.shape[0]: raise ValueError("misaligned matrix sizes: " + str(A.shape) + ", " + str(B.shape))
    mul = np.zeros((A.shape[0], B.shape[1]))
    for i in range(0, mul.shape[0]):
        for j in range(0, mul.shape[1]):
            for k in range(0, A.shape[1]):
                mul[i,j] += A[i,k] * B[k,j]
    return mul

def multimul(mats):
    prod = mul(mats[0], mats[1])
    for i in range(2, len(mats)):
        prod = mul(prod, mats[i])
    return prod
