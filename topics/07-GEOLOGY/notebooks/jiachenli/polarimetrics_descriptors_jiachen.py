import numpy as np

# --------------------- Calculate H/A/alpha ---------------------
"""
Compute H, A, alpha from QP covariance matrix C3. (numpy)
    Input: C3/T3 ([H, W, 3, 3]) - covariance matrix
    Output: H ([H, W]) - entropy in [0, 1]
            A ([H, W]) - anisotropy in [0, 1]
            alpha ([H, W]) - mean alpha angle in degrees [0, 90]
            alpha_stack ([H, W, 3]) - alpha 1, 2, 3 in degrees [0, 90]
            eigvals_map ([H, W, 3]) - eigenvalues of all pixels
"""
def compute_H_A_alpha(C3, eps=1e-10):
    # Stack covariance matrix
    rows, cols = C3.shape[0], C3.shape[1]

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(C3)
    lambda1, lambda2, lambda3 = eigvals[:, :, 2], eigvals[:, :, 1], eigvals[:, :, 0]

    # Compute probabilities
    pr1 = lambda1 / (lambda1 + lambda2 + lambda3 + eps)
    pr2 = lambda2 / (lambda1 + lambda2 + lambda3 + eps)
    pr3 = lambda3 / (lambda1 + lambda2 + lambda3 + eps)
    pr1 = np.clip(pr1, eps, 1.0)
    pr2 = np.clip(pr2, eps, 1.0)
    pr3 = np.clip(pr3, eps, 1.0)

    # Compute entropy
    H = - (pr1 * np.log10(pr1) / np.log10(3) + pr2 * np.log10(pr2) / np.log10(3) + 
                 pr3 * np.log10(pr3) / np.log10(3))
    
    # Compute anisotropy
    A = (lambda2 - lambda3) / (lambda2 + lambda3 + eps)

    # Compute alpha
    U1, U2, U3 = eigvecs[:, :, :, 2], eigvecs[:, :, :, 1], eigvecs[:, :, :, 0]

    alpha_stack = np.zeros((rows, cols, 3), dtype=np.float32) 
    alpha1 = np.arccos(np.abs(U1[:,:,0]))
    alpha2 = np.arccos(np.abs(U2[:,:,0]))
    alpha3 = np.arccos(np.abs(U3[:,:,0]))
    alpha_stack[:, :, 0] = alpha1
    alpha_stack[:, :, 1] = alpha2
    alpha_stack[:, :, 2] = alpha3

    # Mean alpha angle
    alpha = pr1 * alpha1 + pr2 * alpha2 + pr3 * alpha3
    alpha = alpha * 180 / np.pi  # [deg]
    alpha_stack = alpha_stack * 180 / np.pi

    return H, A, alpha, alpha_stack, eigvals

