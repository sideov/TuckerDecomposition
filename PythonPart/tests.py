import unittest
import numpy as np
from numpy.testing import assert_allclose
from main import *

class TestSVDFromScratch(unittest.TestCase):

    def test_svd_from_scratch(self):

        # Test 1
        matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        U1, Sigma1, Vt1 = svd_from_scratch(matrix1)
        self.assertTrue(np.allclose(U1 @ Sigma1 @ Vt1, matrix1, rtol=1e-5, atol=1e-8))


        # Test 2
        matrix2 = np.random.rand(10, 20)
        U2, Sigma2, Vt2 = svd_from_scratch(matrix2)
        self.assertTrue(np.allclose(U2 @ Sigma2 @ Vt2, matrix2, rtol=1e-3, atol=1e-3))


        # Test 3
        matrix3 = np.random.rand(100, 28)
        U3, Sigma3, Vt3 = svd_from_scratch(matrix3)
        self.assertTrue(np.allclose(U3 @ Sigma3 @ Vt3, matrix3, rtol=1e-5, atol=1e-8))



class TestHOOI(unittest.TestCase):

    def valid(self, G, As, X):
        _X = As[1]
        for i in np.arange(2, len(As)):
            _X = np.kron(_X, As[i])

        return np.allclose(refold(As[0]@unfold(G, 0)@_X.T, 0, X.shape), X)

    def test_hooi_decomposition(self):
        for i in range(10):
            X = np.arange(2*3*4*3).reshape(4, 3, 2, 3)
            G, As =  HOOI(X)
            self.valid(G, As, X)


    def test_hooi_diag(self):
        K = 10
        X = np.zeros((K,K,K))

        for i in range(K):
            X[i,i,i] = 1

        G, As = HOOI(X)
        self.valid(G, As, X)

        ans = As[0]
        for i in As[1:]:
            ans = ans @ i

        ans = ans @ G
        print(ans) #OK


if __name__ == '__main__':
    # unittest.main()
    print(np.diag(3,5))
