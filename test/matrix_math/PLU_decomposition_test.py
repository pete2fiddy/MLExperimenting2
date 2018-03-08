import numpy as np
import linalg.matrix.matmath as matmath


N_TESTS = 1000

def test():
    for test_num in range(0, N_TESTS):
        to_decompose = (10*(np.random.rand(8,8)-.5)).astype(np.int).astype(np.float64)
        print("TEST <" + str(test_num) + ">")
        P,L,U = None, None, None
        P, L, U = matmath.PLU_decompose(to_decompose)
        try:
            P, L, U = matmath.PLU_decompose(to_decompose)
        except:
            print("")
        if P is not None:
            assert matmath.is_upper_triangular(U)
            assert matmath.is_lower_triangular(L)
            np.testing.assert_array_almost_equal(to_decompose, np.dot(np.dot(P, L), U))
    print("Test successful")
