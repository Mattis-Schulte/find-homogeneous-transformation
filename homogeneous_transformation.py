import numpy as np

# Given matrices
old_matrix = np.array([[3, -2, 0], [-3, -2, 0], [-5, -7, 0]])
new_matrix = np.array([[3, -2, 3], [-3, -2, 3], [-5, -6, 6]])

# Compute the means of the old and new matrices
old_mean = np.mean(old_matrix, axis=0)
new_mean = np.mean(new_matrix, axis=0)

# De-mean the matrices
old_matrix_demean = old_matrix - old_mean
new_matrix_demean = new_matrix - new_mean

# Compute the rotation matrix using SVD
H = old_matrix_demean.T @ new_matrix_demean
U, S, Vt = np.linalg.svd(H)
R = Vt.T @ U.T

# Compute the translation vector
t = new_mean - R @ old_mean

# Construct the homogeneous transformation matrix
homogeneous_transformation_matrix = np.eye(4)
homogeneous_transformation_matrix[:3, :3] = R
homogeneous_transformation_matrix[:3, 3] = t
print(np.round(homogeneous_transformation_matrix, 2))

# Apply the transformation matrix to the old matrix
transformed_matrix = homogeneous_transformation_matrix @ np.vstack((old_matrix.T, np.ones(old_matrix.shape[0])))
transformed_matrix = transformed_matrix[:3].T
print(np.round(transformed_matrix, 2))
