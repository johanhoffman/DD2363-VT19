import unittest
from lab1 import *

#unit tests for lab 1
class TestMatrixVectorFunctions(unittest.TestCase):

    def test_inner_product(self):
        x = np.array([1,2,3,4])
        y = np.array([1,2,3,4])
        self.assertEquals(inner_product(x,y),np.inner(x,y))
        x = np.array([0,0,0,0])
        self.assertEquals(inner_product(x,y),0)
        z = np.array([1,2,3])
        self.assertEquals(inner_product(x,z),None)

    def test_matrix_vec_prod(self):
        x = np.array([1,2,3,4])
        I = np.array([  [1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]
                    ])
        self.assertEquals(matrix_vec_prod(I,x).tolist(),x.tolist())
        self.assertEquals(matrix_vec_prod(I,np.array([1,2,3])), None)

        b = np.array([1,2,3])
        A = np.array([  [1,20,3],
                        [-2,1,4],
                        [9,-12,0],
                        [1,1,6]
                    ])
        self.assertEquals(matrix_vec_prod(A,b).tolist(),A.dot(b).tolist())
        self.assertEquals(matrix_vec_prod(A,x),None)

    def test_matrix_matrix_prod(self):
        I = np.array([  [1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]
                    ])
        A = np.array([  [1,20,3],
                        [-2,1,4],
                        [9,-12,0],
                        [1,1,6]
                    ])
        self.assertEquals(matrix_matrix_prod(A,I), None)
        B = np.array([  [1,2,3,4],
                        [0,2,4,-2],
                        [-2,9,6,1],
                        [0,4,3,2]
                    ])
        self.assertEquals(matrix_matrix_prod(B,I).tolist(),matrix_matrix_prod(I,B).tolist())
        self.assertEquals(matrix_matrix_prod(B,I).tolist(),B.tolist())
        self.assertEquals(matrix_matrix_prod(B,A).tolist(),np.matmul(B,A).tolist())
        self.assertEquals(matrix_matrix_prod(A,B), None)

    def test_SparseMatrix_class(self):
        val = np.array([1,2,3,4,5])
        col_idx = np.array([0,1,2,3,4])
        row_ptr = np.array([0,1,2,3,4,5])
        A = SparseMatrix(val,col_idx,row_ptr)
        self.assertEquals(A.val.tolist(),val.tolist())
        self.assertEquals(A.col_idx.tolist(),col_idx.tolist())
        self.assertEquals(A.row_ptr.tolist(),row_ptr.tolist())

    def test_SparseMatrix_vec_prod(self):
        val = np.array([4,2,3,4,5,-2,6,2])
        col_idx = np.array([0,1,1,4,1,2,3,4])
        row_ptr = np.array([0,0,2,4,4,8])
        A_sparse = SparseMatrix(val,col_idx,row_ptr)
        A = np.array([
            [0,0,0,0,0],
            [4,2,0,0,0],
            [0,3,0,0,4],
            [0,0,0,0,0],
            [0,5,-2,6,2]
        ])
        I_sparse = SparseMatrix(np.array([1,1,1,1,1]), np.array([0,1,2,3,4]), np.array([0,1,2,3,4,5]))
        I = np.array([
            [1,0,0,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [0,0,0,0,1]
        ])
        x = np.array([-2,3,7,10,1])
        print SparseMatrix_vec_prod(A_sparse,x)
        self.assertEquals(SparseMatrix_vec_prod(A_sparse,x).tolist(),A.dot(x).tolist())
        self.assertEquals(SparseMatrix_vec_prod(I_sparse,x).tolist(),x.tolist())
if __name__ == '__main__':
    unittest.main()
