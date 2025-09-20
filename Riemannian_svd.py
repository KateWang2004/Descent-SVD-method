import numpy as np
import time
import gc
import os
import psutil

# Utility to track memory usage
def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024**2  # in MB

# Objective function F
def F(P, Q, A, Sigma):
    """Objective function F"""
    # Calculate residual
    residual = P.T @ A @ Q @ Sigma  # P^T * A * Q * Sigma

    # Return the objective function value using the trace (tr)
    return -np.trace(residual)  # Calculate and return the objective function value

# Symmetric part of a matrix
def sym(X):
    """Compute the symmetric part sym(X)"""
    return (X + X.T) / 2

# Compute the Riemannian gradient of the objective function F
def compute_grad_F(P, Q, A, N):
    """
    Compute the Riemannian gradient of the objective function F(P, Q)
    :param P: Left factor matrix P
    :param Q: Right factor matrix Q
    :param N: Weighting matrix (diagonal matrix)
    :return: Riemannian gradient (grad_P, grad_Q)
    """

    # Compute the symmetric parts S1 and S2
    S1 = sym(np.dot(P.T, np.dot(A, np.dot(Q, N))))  # P^T * A * Q
    S2 = sym(np.dot(Q.T, np.dot(A.T, np.dot(P, N))))  # Q^T * Y_Q^T * P
    
    # Compute the Riemannian gradient
    grad_P = np.dot(P, S1) - np.dot(A, np.dot(Q, N))
    grad_Q = np.dot(Q, S2) - np.dot(A.T, np.dot(P, N))
    
    return grad_P, grad_Q

# Stiefel manifold retraction operation
def stiefel_retraction(P, direction_P, alpha):
    """Perform retraction on the Stiefel manifold"""
    P_new = P + alpha * direction_P  # Gradient descent update
    # Ensure P_new is still orthogonal
    Q, R = np.linalg.qr(P_new)  # QR decomposition to maintain orthogonality
    return Q

# Wolfe step size computation
def wolfe_step_size(P, Q, direction_P, direction_Q, grad_P, grad_Q, A, Sigma, sigma=1e-5, rho=0.8, alpha_init=1.0):
    """Compute step size that satisfies Wolfe conditions"""
    
    alpha_k = alpha_init  # Initialize step size
    
    f_PQ = F(P, Q, A, Sigma)  # Compute initial objective function value
    inner_prod = np.dot(grad_P.flatten(), direction_P.flatten()) + np.dot(grad_Q.flatten(), direction_Q.flatten())
    
    while alpha_k >= 1e-6:  # Prevent the step size from becoming too small
        new_P = stiefel_retraction(P, direction_P, alpha_k)  # Perform retraction update on P
        new_Q = stiefel_retraction(Q, direction_Q, alpha_k)  # Perform retraction update on Q
        
        f_new = F(new_P, new_Q, A, Sigma)  # Compute new objective function value
        new_grad_P, new_grad_Q = compute_grad_F(new_P, new_Q, A, Sigma)  # Compute new gradients

        # Check the Armijo condition (Equation 42)
        if f_PQ - f_new >= -sigma * alpha_k * inner_prod:
            # Check the curvature condition (Equation 43)
            transported_xi, transported_eta = vector_transport(P, Q, direction_P, direction_Q, direction_P, direction_Q, alpha_k)
            curvature_condition = np.dot(new_grad_P.flatten(), transported_xi.flatten()) + np.dot(new_grad_Q.flatten(), transported_eta.flatten())
            if curvature_condition >= rho * inner_prod:
                return alpha_k
        
        # If conditions are not met, reduce step size
        alpha_k *= 0.8  # Reduce the step size
    
    return alpha_k

# Skew-symmetric part of a matrix
def skewh(X):
    """Compute the skew-symmetric part of a matrix"""
    n = X.shape[0]
    skewh_X = np.zeros_like(X)
    
    for i in range(n):
        for j in range(n):
            if i > j:
                skewh_X[i, j] = X[i, j]  # Upper triangular part
            elif i < j:
                skewh_X[i, j] = -X[i, j]  # Lower triangular part
            else:
                skewh_X[i, j] = 0  # Diagonal elements set to 0
    
    return skewh_X

# Compute the derivative Dqf(X)[Z]
def Dqf(X, Z):
    """Compute Dqf(X)[Z], the derivative of the function"""
    U, R = np.linalg.qr(X)  # QR decomposition
    R_inv = np.linalg.inv(R)
    temp = np.dot(U.T, np.dot(Z, R_inv))  # U^* Z R^-1
    
    skewh_part = np.dot(U, skewh(temp))  # U * skewh(U^* Z R^-1)
    I_n = np.eye(U.shape[0])  # Identity matrix I_n
    projection_part = np.dot((I_n - np.dot(U, U.T)), np.dot(Z, R_inv))  # (I_n - U U^*) Z R^-1
    
    return skewh_part + projection_part

# Vector transport implementation (Equation 31)
def vector_transport(P, Q, xi, eta, x, y, alpha):
    """Perform vector transport on the Stiefel manifold"""
    P_new = P + alpha * xi
    Q_new = Q + alpha * eta

    U_P, R_P = np.linalg.qr(P_new)
    U_Q, R_Q = np.linalg.qr(Q_new)

    sym_P = sym(U_P.T @ x)
    sym_Q = sym(U_Q.T @ y)

    transported_xi = x - U_P @ sym_P
    transported_eta = y - U_Q @ sym_Q

    return transported_xi, transported_eta

# Compute the parameter s_k based on the transported vectors
def compute_sk(xi_k, eta_k, transported_xi, transported_eta):
    """Compute parameter s_k, decomposed into P and Q components"""
    combined = np.concatenate((xi_k.flatten(), eta_k.flatten()))  # Combine xi_k and eta_k
    transported_combined = np.concatenate((transported_xi.flatten(), transported_eta.flatten()))
    s_k = min(1, np.linalg.norm(combined) / np.linalg.norm(transported_combined))  # Compute s_k
    
    return s_k

# Riemannian conjugate gradient method (Stiefel manifold optimization)
def riemannian_cg(A, Sigma, p, max_iter=50000, tol=20., tol_f=1e-20):
    """Conjugate gradient optimization on the Stiefel manifold"""
    m, n = A.shape
    P, s, Q = np.linalg.svd(A, full_matrices=False)  # SVD decomposition
    
    P = np.random.randn(m, p)  # Randomly initialize P
    Q = np.random.randn(n, p)  # Randomly initialize Q

    grad_P, grad_Q = compute_grad_F(P, Q, A, Sigma)
    direction_P = -grad_P
    direction_Q = -grad_Q

    for k in range(max_iter):
        alpha = wolfe_step_size(P, Q, direction_P, direction_Q, grad_P, grad_Q, A, Sigma)
        new_P = stiefel_retraction(P, direction_P, alpha)
        new_Q = stiefel_retraction(Q, direction_Q, alpha)

        grad_P_new, grad_Q_new = compute_grad_F(new_P, new_Q, A, Sigma)

        grad_combined = np.concatenate((grad_P.flatten(), grad_Q.flatten()))  # Combine gradients
        grad_combined_new = np.concatenate((grad_P_new.flatten(), grad_Q_new.flatten()))  # Combine new gradients
        a, b = vector_transport(P, Q, direction_P, direction_Q, grad_P, grad_Q, alpha)
        grad_combined_cha = np.concatenate((a.flatten(), b.flatten()))  # Combine gradient changes

        beta = np.dot(grad_combined_new, (grad_combined_cha - grad_combined_cha)) / np.dot(grad_combined, grad_combined)

        transported_xi, transported_eta = vector_transport(P, Q, direction_P, direction_Q, direction_P, direction_Q, alpha)
        direction_P = -grad_P_new + beta * transported_xi
        direction_Q = -grad_Q_new + beta * transported_eta

        grad_norm = np.linalg.norm(grad_combined_new)  # Compute gradient norm

        if k % 100 == 0:
            print(f'Iteration: {k}')
            print(f'Gradient norm: {grad_norm}')
            print(f'Objective function value: {F(P, Q, A, Sigma)}')
            print(f'Step size: {alpha}')

        tol_pq = np.linalg.norm(new_P - P) / np.sqrt(m) + np.linalg.norm(new_Q - Q) / np.sqrt(n)
        F_new = F(new_P, new_Q, A, Sigma)
        F_old = F(P, Q, A, Sigma)
        tol_F = np.abs(F_new - F_old) / (np.abs(F_old) + 1)

        if grad_norm < tol or tol_pq < tol_f or tol_F < tol_f:
            print(f"Converged at iteration {k}")
            print(f"tol_pq: {tol_pq}, tol_F: {tol_F}, grad_norm: {grad_norm}")
            break

        P, Q = new_P, new_Q
        grad_P, grad_Q = grad_P_new, grad_Q_new

    return P, Q

# Generate a matrix with random orthonormal matrices
def generate_matrix(m, n):
    """Generate matrix A = U_r diag(σ_1, ..., σ_n) V_r^T"""
    np.random.seed(42)
    
    U_r = np.random.randn(m, n)
    Q, _ = np.linalg.qr(U_r)
    
    V_r = np.random.randn(n, n)
    Q_V, _ = np.linalg.qr(V_r)
    
    sigma = np.random.uniform(0, 100, size=n)
    sigma = np.sort(sigma)[::-1]
    
    Sigma = np.diag(sigma)
    
    A = np.dot(Q, np.dot(Sigma, Q_V.T))
    
    return A


def sigma_get(P, A, Q):
    """
    Compute a matrix generated by the outer products of the first row of P, 
    the first column of A, and the first column of Q.
    :param P: Matrix P (m x p)
    :param A: Matrix A (m x n)
    :param Q: Matrix Q (n x p)
    :return: A new matrix generated by the outer product.
    """
    # Initialize the sigma matrix with zeros (same dimensions as P)
    sigma_matrix = np.zeros((P.shape[0], P.shape[0]))

    # Loop over each row of P to fill the diagonal of sigma_matrix
    for i in range(P.shape[0]):
        row_P = P[i, :]  # Extract the i-th row of P
        col_Q = Q[:, i]  # Extract the i-th column of Q
        # Compute the value for the diagonal element
        sigma_matrix[i, i] = row_P @ A @ col_Q  # Dot product of row_P, A, and col_Q

    return sigma_matrix


# Test the optimization algorithm
if __name__ == "__main__":
    A = generate_matrix(30, 10)  # Generate a random matrix
    np.save('./matrix_stiefel/generated_matrix_300_20.npy', A)
    
    p = 5  # Rank truncation
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    Sigma = np.diag(s[-p:])
    Up = U[:, :p]
    Vp = Vh[:p, :]

    print("Objective function value:", F(A, Sigma, Up, Vp))
    
    time1 = time.time()
    gc.collect()
    mem1 = get_memory_usage_mb()
    
    P, Q = riemannian_cg(A, Sigma, p)
    
    print(f'Optimization time: {time.time() - time1:.2f} seconds')
    print(f'Memory usage after optimization: {get_memory_usage_mb() - mem1:.2f} MB')
    

    Sigma_get = sigma_get(P.T, A, Q)
    

    print(f'Reconstruction accuracy: {1. - np.linalg.norm(A - P @ Sigma_get @ Q.T) / np.linalg.norm(A)}')

    print(f'P^T P diff from I: {np.linalg.norm(P.T @ P - np.eye(P.shape[1]))}')  # Check orthogonality of P
    print(f'Q^T Q diff from I: {np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))}')  # Check orthogonality of Q
