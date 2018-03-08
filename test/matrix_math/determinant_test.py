import numpy as np
import linalg.matrix.matmath as matmath

N_TESTS = 1000

def test():
    for test_num in range(N_TESTS):
        to_det = (10*(np.random.rand(5,5)-.5)).astype(np.float64)
        print("Test <" + str(test_num) + ">")
        np.testing.assert_approx_equal(np.linalg.det(to_det), matmath.det(to_det), significant = 7)
        #assert float(np.linalg.det(to_det)) == float(matmath.det(to_det)), "correct det: " + str(np.linalg.det(to_det)) + ", answered det: " + str(matmath.det(to_det))
