import torch
import numpy as np
import time
import gc
from PIL import Image
import os
import psutil

# Utility to track memory usage
def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024**2  # in MB

# Read image as grayscale and print its shape
def read_image_as_gray_and_print_shape(image_path):

    img = Image.open(image_path).convert('L')
    img_np = np.array(img, dtype=np.float32)
    print(f"Image shape: {img_np.shape}")
    img_tensor = torch.from_numpy(img_np).float()
    return img_tensor

# Randomized SVD Method
def randomized_svd(A, k=100, n_iter=10, device="cpu"):
    """
    Randomized SVD method

    Parameters:
      A: Input matrix, shape (m, n)
      k: Target rank, compute the first k singular values/vectors
      n_iter: Number of power iterations to improve the approximation accuracy
      device: Computation device ("cpu" or "cuda")

    Returns:
      U: Approximate left singular vectors, shape (m, k)
      S: Approximate singular value diagonal matrix, shape (k, k)
      VT: Approximate transpose of the right singular vectors, shape (k, n)
    """
    A = A.to(device)
    m, n = A.shape

    # Generate random projection matrix Omega (n x k)
    Omega = torch.randn(n, k, device=device)
    Y = A @ Omega  # Y = A * Omega (m x k)

    # QR decomposition on Y
    Q, _ = torch.linalg.qr(Y, mode="reduced")

    # Power iterations to improve approximation accuracy
    for _ in range(n_iter):
        Y = A.T @ Q
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        Y = A @ Q
        Q, _ = torch.linalg.qr(Y, mode="reduced")

    # Construct small matrix B = Qᵀ * A
    B = Q.T @ A
    U_B, S_vec, VT = qr_jacobi_svd(B.T, device=device)

    # Final approximation of U and S
    U_B_ = VT.T
    VT = U_B.T
    U_B = U_B_

    # Compute U = Q * U_B
    U = Q @ U_B
    S = S_vec

    return U, S, VT.T

# QR-Jacobi SVD Method
def qr_jacobi_svd(A, max_iter=100, tol=1e-4, device="cpu"):
    """
    Compute SVD using QR-Jacobi method

    Parameters:
      A: Input matrix (m, n)
      max_iter, tol: Parameters for QR-Jacobi iterations
      device: Computation device ("cpu" or "cuda")

    Returns:
      U_final: m×n left singular vector matrix
      S_matrix: n×n diagonal singular value matrix
      VT: n×n right singular vector transpose
    """
    A = A.to(device)
    m, n = A.shape

    # QR decomposition, A = Q * R
    Q, R = torch.linalg.qr(A, mode="reduced")

    # Jacobi SVD on R
    S_matrix, U, V = lanczos_svd_torch(R, k=n, device=device, max_iter=max_iter)

    # Final left singular vectors U = Q * U_r
    U_final = Q @ U

    return U_final, S_matrix, V.T

# Lanczos SVD Method
def lanczos_svd_torch(matrix, k, device, max_iter=1000):
    """
    Compute the top k singular values of a matrix using Lanczos method.

    Parameters:
        matrix: Input matrix (n, m)
        k: Number of singular values to compute
        device: Computation device ("cpu" or "cuda")
        max_iter: Maximum number of Lanczos iterations

    Returns:
        singular_values: Top k singular values of the matrix
        U_approx: Approximate left singular vectors
        V_approx: Approximate right singular vectors
    """
    matrix = matrix.to(device)
    n, m = matrix.shape
    max_iter = min(max_iter, m)

    # Initialize Lanczos vectors and iteration variables
    v = torch.randn(m, device=device)
    v = v / torch.norm(v)
    V = torch.zeros((max_iter, m), device=device)
    alpha = torch.zeros(max_iter, device=device)
    beta = torch.zeros(max_iter, device=device)

    # Lanczos iterations
    for j in range(max_iter):
        V[j] = v
        w = matrix.T @ (matrix @ v)
        alpha[j] = torch.dot(w, v)
        w = w - alpha[j] * v - (beta[j-1] * V[j-1] if j > 0 else 0)
        
        # Re-orthogonalize
        for i in range(j):
            w -= torch.dot(w, V[i]) * V[i]
        
        beta[j] = torch.norm(w)
        if beta[j] < 1e-10:
            break
        
        v = w / (beta[j] + 1e-12)

    # Construct the tridiagonal matrix T
    T = torch.diag(alpha[:j+1]) + torch.diag(beta[:j], diagonal=1) + torch.diag(beta[:j], diagonal=-1)
    eigvals, eigvec = torch.linalg.eigh(T)
    eigvals = torch.clamp(eigvals, min=1e-12)
    singular_values = torch.sqrt(eigvals)

    # Approximate right singular vectors (V)
    V_approx = V[:j+1].T @ eigvec
    V_approx = V_approx / torch.norm(V_approx, dim=0, keepdim=True)

    # Approximate left singular vectors (U)
    U_approx = matrix @ V_approx
    U_approx = U_approx / (torch.norm(U_approx, dim=0, keepdim=True) + 1e-12)

    return singular_values[-k:], U_approx[:, -k:], V_approx[:, -k:]

# Generate matrix with almost equal singular values
def generate_matrix_with_almost_equal_singular_values(m, n, value=1.0):
    """
    生成一个所有奇异值都接近的矩阵。
    """
    A = torch.randn(m, n)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    max_s = S.max().item()
    S_new = S.clone()
    epsilon = 1e-6
    S_new[:10] = max_s * (1 - epsilon + epsilon * torch.rand(10))
    
    A_new = (U @ torch.diag(S_new) @ Vh)
    return A_new

# Generate matrix with singular value decay
def generate_matrix_with_singular_value_decay(m, n, mode='exp', value=1.0, k_trunc=5, power=2, poly_coeff=1.0, exp_rate=0.8):
    """
    生成一个奇异值按指定规律衰减的矩阵。
    """
    A = torch.randn(m, n)
    S_ori = torch.linalg.svdvals(A)
    torch.manual_seed(0)
    min_dim = min(m, n)
    
    if mode == 'exp':
        s = value * torch.exp(-exp_rate * torch.arange(min_dim))
    elif mode == 'power':
        s = value / (torch.arange(1, min_dim+1) ** power)
    elif mode == 'poly':
        s = value / (1 + poly_coeff * torch.arange(min_dim) ** power)
    elif mode == 'trunc':
        s = torch.zeros(min_dim)
        s += 1e-6
        s[:k_trunc] = S_ori[:k_trunc]
    else:
        raise ValueError("Unknown mode")
    
    Q1, _ = np.linalg.qr(np.random.randn(m, m))
    Q2, _ = np.linalg.qr(np.random.randn(n, n))
    S_mat = torch.diag(s)
    A_new = torch.from_numpy(Q1[:, :min_dim] @ S_mat.numpy() @ Q2[:min_dim, :].T).float()
    
    return A_new

# Example: Generate matrix, perform SVD, and measure accuracy
def main():
   
    
    input_matrix = np.load('./matrix_stiefel/generated_matrix_300_20.npy')
    input_tensor = torch.tensor(input_matrix, dtype=torch.float32)
    
    print("Matrix condition number:", torch.linalg.cond(input_tensor))
    print("Matrix rank:", torch.linalg.matrix_rank(input_tensor))
    print("SVD values:", torch.linalg.svdvals(input_tensor))
    
    k = 10
    accuracies = []
    times = []

    for _ in range(10):
        time1 = time.time()
        gc.collect()
        
        U, singular_values, V = randomized_svd(input_tensor, k,)
        reconstructed = (U @ torch.diag(singular_values) @ V.T)
        accuracy = 1. - torch.norm(reconstructed - input_tensor) / torch.norm(input_tensor)
        
        accuracies.append(accuracy.item())
        times.append(time.time() - time1)
    
    # Compute mean and variance
    accuracy_mean = np.mean(accuracies)
    accuracy_variance = np.var(accuracies)
    time_mean = np.mean(times)
    time_variance = np.sqrt(np.var(times))

    print(f"Mean accuracy: {accuracy_mean:.4f}, Accuracy variance: {accuracy_variance:.10f}")
    print(f"Mean time: {time_mean:.4f} seconds, Time variance: {time_variance:.4f} seconds")

if __name__ == "__main__":
    main()
