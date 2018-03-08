import numpy as np
import linalg.matrix.matmath as matmath

N_TESTS = 1000

def test():
    for test_num in range(N_TESTS):
        print("Test <" + str(test_num) + ">")
        try:
            to_inv = (10*(np.random.rand(50, 50)-0.5)).astype(np.float64)
            np.testing.assert_array_almost_equal(np.linalg.inv(to_inv), matmath.inv(to_inv))
        except ValueError:
            print("Singular matrix generated. Skipping")
