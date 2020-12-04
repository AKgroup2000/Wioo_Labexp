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


# Without Numpy:


def print_matrix(Title, M):
    print(Title)
    for row in M:
        print([round(x,3)+0 for x in row])
        
def print_matrices(Action, Title1, M1, Title2, M2):
    print(Action)
    print(Title1, '\t'*int(len(M1)/2)+"\t"*len(M1), Title2)
    for i in range(len(M1)):
        row1 = ['{0:+7.3f}'.format(x) for x in M1[i]]
        row2 = ['{0:+7.3f}'.format(x) for x in M2[i]]
        print(row1,'\t', row2)
        
def zeros_matrix(rows, cols):
    A = []
    for i in range(rows):
        A.append([])
        for j in range(cols):
            A[-1].append(0.0)

    return A

def copy_matrix(M):
    rows = len(M)
    cols = len(M[0])

    MC = zeros_matrix(rows, cols)

    for i in range(rows):
        for j in range(rows):
            MC[i][j] = M[i][j]

    return MC

def matrix_multiply(A,B):
    rowsA = len(A)
    colsA = len(A[0])

    rowsB = len(B)
    colsB = len(B[0])

    if colsA != rowsB:
        print('Number of A columns must equal number of B rows.')
        sys.exit()

    C = zeros_matrix(rowsA, colsB)

    for i in range(rowsA):
        for j in range(colsB):
            total = 0
            for ii in range(colsA):
                total += A[i][ii] * B[ii][j]
            C[i][j] = total

    return C

A = [[5,4,3,2,1],[4,3,2,1,5],[3,2,9,5,4],[2,1,5,4,3],[1,2,3,4,5]]
I = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
print_matrix('A Matrix', A)
print()
print_matrix('I Matrix', I)

*********
# Matrix solution findind of AX =B
'''
Solution of 
Ax =B 
x = A^-1 B '''
import numpy as np
def Inverse(M):
  Minv = np.linalg.inv(M)
  return Minv

A = [[1,0,0],
    [1,1,1],
    [6,7,0]]
b = [0,24,0]
# Find inverse of matrix A.
Ai = Inverse(A) 
x = np.dot(Ai,b)
print("Resut = ",x)
******
# RANK of a matrix:

from numpy import matrix
A = matrix([[1,3,7],[2,8,3]], dtype=float)  # doesn't accept int

import scipy.linalg.interpolative as sli
rank = sli.estimate_rank(A, eps=1e-10)
print("Rank = ", rank)

