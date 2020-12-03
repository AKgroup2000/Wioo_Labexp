'''
Creating Matrices
We can pass python lists of lists in the following shape to have NumPy create a matrix to represent them:
np.array([[1,2],[3,4]])'''

import numpy as np
a = np.array([[1,2,2],[2,3,4]])
print("2D array A \n",a)

******

'''
The inverse of a matrix exists only if the matrix is non-singular
i.e., determinant should not be 0. Using determinant and adjoint, we can easily find the inverse of a square matrix using below formula,
if det(A) != 0
    A-1 = adj(A)/det(A)
else
    "Inverse doesn't exist"  
'''
# The function numpy.linalg.inv() which is available in the python NumPy module is used to compute the inverse of a matrix. 
import numpy as np
M = np.array([[1,2,0],[3,4,0],[5,6,1]])
Minv = np.linalg.inv(M)
print("Inversae of M : \n",Minv)



