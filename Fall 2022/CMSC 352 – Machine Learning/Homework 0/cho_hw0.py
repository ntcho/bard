import numpy as np

mat = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

vec = np.array([[4], 
                [2], 
                [0]])

print(np.matmul(mat, vec))
