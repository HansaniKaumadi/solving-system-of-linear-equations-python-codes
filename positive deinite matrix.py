import numpy as np
from scipy.linalg import cholesky

def is_positive_definite(matrix):
    # Check if all eigenvalues are positive
    eigenvalues = np.linalg.eigvals(matrix)
    if np.all(eigenvalues > 0):
        return True
    
    # Check if the matrix is positive definite using Cholesky decomposition
    try:
        _ = cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

# Example usage
matrix = np.array([[2, -1, 0],
                   [-1, 2, -1],
                   [0, -1, 2]])

print(is_positive_definite(matrix))  # Output: True or False