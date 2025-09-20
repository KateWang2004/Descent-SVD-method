import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import functional
import matplotlib.pyplot as plt
from PIL import Image
import time
import numpy as np
from mpi4py import MPI 
from utils import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def randomized_svd(A, k, n_max, epsilon, n_iter=20, device="cpu"):
    """
    Randomized SVD method.
    Parameters:
        A: Input matrix of shape (m, n)
        k: Target rank for SVD
        n_max: Maximum number of iterations for SVD computation 
        epsilon: Tolerance for convergence
        n_iter: Number of power iterations to improve approximation accuracy (default is 1, can be increased to 2~3)
        device: Device for computation ("cpu" or "cuda")
    Returns:
        U: Approximate left singular vectors of shape (m, k)
        S: Approximate singular values diagonal matrix of shape (k, k)
        VT: Approximate right singular vectors transpose of shape (k, n
    """

    # Move A to the target device
    A = A.to(device)
    m, n = A.shape
    
    if rank == 0:
        # 1. Generate a random projection matrix Omega (n x k)
        Omega = torch.randn(n, k, device=device)
        # 2. Compute Y = A * Omega (m x k)
        Y = A @ Omega
        # 3. Perform QR decomposition on Y to get orthogonal matrix Q (m x k)
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        
        # # Optionally, perform several power iterations to improve approximation accuracy
        for _ in range(n_iter):
            Y = A.T @ Q  # (n x k)
            Q, _ = torch.linalg.qr(Y, mode="reduced")
            Y = A @ Q    # (m x k)
            Q, _ = torch.linalg.qr(Y, mode="reduced")

        B = Q.T @ A
    else: 
        B = None

    B = comm.bcast(B, root=0)
    comm.Barrier()
    res = qr_jacobi_svd(B.T, n_max, epsilon, device=device)
    if res is None:
        return None
    
    U_B, S_vec, VT = res
    U_B_ = VT.T
    VT = U_B.T
    U_B = U_B_
    # # 6. Compute U = Q * U_B, shape (m x k)
    U = Q @ U_B
    S = S_vec
    
    return U, S, VT


def qr_jacobi_svd(A, n_max, epsilon, max_iter=1000, tol=1e-6, device="cpu"):
    """
    Use QR-Jacobi method to compute SVD, supports CPU and CUDA.
    Assume A is an m×n matrix (m >= n), return reduced SVD:
      A ≈ U * S * VT, where U is m×n, S is n×n diagonal matrix, VT is n×n.
    
    Parameters:
      A: Input matrix (m, n)
      max_iter, tol: Parameters passed to Jacobi iteration (used for SVD computation of R)
      device: Computation device ("cpu" or "cuda")
    
    Returns:
      U_final: m×n left singular vector matrix
      S_matrix: n×n diagonal singular value matrix
      VT: n×n right singular vector transpose
    """
    A = A.to(device)
    m, n = A.shape
    
    # Step 1: QR decomposition to get A = Q * R
    if rank == 0:
        Q, R = torch.linalg.qr(A, mode="reduced")  # Q: m×n, R: n×n
    else:
        Q = None
        R = None

    Q = comm.bcast(Q, root=0)
    R = comm.bcast(R, root=0)

    res = parallel_svd(R, R.shape[0], n_max, epsilon, device=device)
    if res is None:
        return None
    
    U_r, S_matrix, VT  = res
    # Step 3: Final left singular vector U = Q * U_r
    U_final = Q @ U_r  # m×n

    return U_final, S_matrix, VT


def process_columns_batch(batch_indices, a, Phi, Psi, W, V, E, R, K, D, N, M, n_max, epsilon, gamma_diag, device):
    """
    Process a batch of columns in parallel.
    Parameters:
        batch_indices: Indices of the columns to process
        a: Number of rows in the matrix
        Phi: Transformed matrix
        Psi: Transformed matrix
        W, V, E, R: Matrices to be updated
        K, D, N, M: Dimensions of the matrices
        n_max: Maximum number of iterations for optimization
        epsilon: Tolerance for convergence
        gamma_diag: Diagonal elements of the gamma matrix
        device: Device for computation ("cpu" or "cuda")
    """
    start_time1 = time.time()
    batch_size = len(batch_indices)
    batch_size = 1
    column_index = batch_indices[0]
    
    # Initialize column data for the current batch
    w = W[:, column_index: column_index+batch_size].to(device).reshape(D, batch_size)
    v = V[:, column_index: column_index+batch_size].to(device).reshape(D, batch_size)
    e = E[column_index:column_index+batch_size, :].to(device).reshape(batch_size, N)
    r = R[column_index:column_index+batch_size, :].to(device).reshape(batch_size, M)
    gamma = torch.diag(gamma_diag[column_index:column_index+batch_size]).to(device)
    
    # Construct matrix A and vector b
    time_constr = time.time()
    A, b = construct_A_and_b(Phi, Psi, D, batch_size, N, M, device)
    v_ = torch.randn(A.shape[0]).to(device)
    params = [w, v, e, r]
    
    # Main iterative optimization loop
    for iteration in range(n_max):
        def loss_func(p, Phi, Psi, gamma, batch_size):
            w, v, e, r = p[:D*batch_size], p[D*batch_size:2*D*batch_size], p[2*D*batch_size:2*D*batch_size + N*batch_size], p[2*D*batch_size + N*batch_size:2*D*batch_size + N*batch_size + M*batch_size]
            w, v, e, r = w.view(D, batch_size), v.view(D, batch_size), e.view(batch_size, N), r.view(batch_size, M)
            return (-torch.trace(torch.matmul(w.T, v).to(device)) +
                        0.5 * torch.trace(torch.matmul(torch.matmul(e.T, gamma).to(device), e)) +
                        0.5 * torch.trace(torch.matmul(torch.matmul(r.T, gamma).to(device), r)))
        
        time_iteration = time.time()
        if iteration == 0:
            grad, hessian = compute_gradient_and_hessian(params, loss_func, (Phi, Psi, gamma, batch_size), False, gamma, device)
        else:
            grad, _ = compute_gradient_and_hessian(params, loss_func, (Phi, Psi, gamma, batch_size), True, gamma, device)
        
        params_tensor = torch.cat([p.reshape(-1) for p in params]).to(device)
        params_tensor.requires_grad_()
        
        # Update parameters using KKT conditions
        time_kkt = time.time()
        params, v_ = update_params_kkt(params, grad, hessian, A, b, v_, device)
        
    # Update the matrices after processing
    time_to_write = time.time()
    W[:, column_index: column_index+batch_size] = w.reshape(-1, batch_size).to(device)
    V[:, column_index: column_index+batch_size] = v.reshape(-1, batch_size).to(device)
    E[column_index:column_index+batch_size, :] = e.reshape(batch_size, -1).to(device)
    R[column_index:column_index+batch_size, :] = r.reshape(batch_size, -1).to(device)


def grad_constrained_newton(matrix, a, phi, psi, gamma_diag, n_max, epsilon, device):
    """
    Gradient Constrained Newton method for SVD.

    Parameters:
        matrix: Input matrix of shape (N, M)
        phi: Function to transform rows of the matrix
        psi: Function to transform columns of the matrix
        gamma_diag: Diagonal elements of the gamma matrix
        n_max: Maximum number of iterations for optimization
        epsilon: Tolerance for convergence
        device: Device for computation ("cpu" or "cuda")
    """
    N, M = matrix.shape
    D = phi(matrix[0, :]).shape[1]
    K = len(gamma_diag)
    
    # Compute Phi and Psi using the phi and psi functions
    Phi = torch.stack([phi(matrix[i, :]) for i in range(N)]).to(device).reshape((N, D))
    Psi = torch.stack([psi(matrix[:, j]) for j in range(M)]).to(device).mT.reshape((M, D))

    matrix_shapes = [
        (D, K),  # W shape
        (D, K),  # V shape
        (K, N),  # E shape
        (K, M)   # R shape
    ]

    num_matrices = len(matrix_shapes)
    dtype = np.float32  # Data type

    # Calculate offsets for each tensor in shared memory
    offsets = [0]
    for shape in matrix_shapes:
        offsets.append(offsets[-1] + np.prod(shape))  # Cumulative size for each tensor
    
    # Only the root process (rank == 0) initializes shared memory
    if rank == 0:
        shared_data = np.zeros(offsets[-1], dtype=dtype)
    else:
        shared_data = None

    # Create shared memory window
    win = MPI.Win.Allocate_shared(offsets[-1] * np.dtype(dtype).itemsize, np.dtype(dtype).itemsize, comm=comm)

    # Allow all processes to access shared memory
    buf, itemsize = win.Shared_query(0)
    shared_data = np.ndarray((offsets[-1],), dtype=dtype, buffer=buf)

    # Map the shared memory to PyTorch tensors
    W = torch.from_numpy(shared_data[offsets[0]:offsets[1]].reshape(matrix_shapes[0])).to(device)
    V = torch.from_numpy(shared_data[offsets[1]:offsets[2]].reshape(matrix_shapes[1])).to(device)
    E = torch.from_numpy(shared_data[offsets[2]:offsets[3]].reshape(matrix_shapes[2])).to(device)
    R = torch.from_numpy(shared_data[offsets[3]:offsets[4]].reshape(matrix_shapes[3])).to(device)

    W.mul_(1e-6)
    V.mul_(1e-6)
    E.mul_(1e-6)
    R.mul_(1e-6)
    
    # Process columns in batches
    batch = [rank]
    process_columns_batch(batch, a, Phi, Psi, W, V, E, R, K, D, N, M, n_max, epsilon, gamma_diag, device)

    time1 = time.time()
    comm.Barrier()

    W = W.clone()
    V = V.clone()
    E = E.clone()
    R = R.clone()

    win.Free()

    if rank == 0:
        return [W, V, E, R]
    else:
        return None


def parallel_svd(matrix, k, n_max, epsilon, device):
    """
    Parallel SVD using descent method.
    Parameters:
        matrix: Input matrix of shape (N, M)
        k: Target rank for SVD
        n_max: Maximum number of iterations for optimization
        epsilon: Tolerance for convergence
        device: Device for computation ("cpu" or "cuda")
    Returns:
        alp: Left singular vectors
        det: Diagonal singular values
        beta: Right singular vectors transpose
    """
    matrix = matrix.to(device)
    N, M = matrix.shape

    if rank == 0:
        # S = lanczos_svd_torch(matrix, k, device)
        S = torch.linalg.svdvals(matrix).to(device)
    else:
        S = None

    S = comm.bcast(S, root=0)

    C = choose_compatible_matrix(matrix).to(device)
   
    def phi(x):
        transformed = C.T @ x.float()
        return transformed.unsqueeze(0) if transformed.dim() == 1 else transformed

    def psi(z):
        return z.unsqueeze(0) if z.dim() == 1 else z

    result = grad_constrained_newton(matrix, N, phi, psi, 1/S, n_max, epsilon, device)
    if result is None:
        return None

    _, _, E, R = result
    E_opt = E
    R_opt = R

    det = create_expanded_diagonal(S, N, M, device)

    alp = torch.nn.functional.normalize(E_opt.data, p=2, dim=1).T.to(device)
    beta = torch.nn.functional.normalize(R_opt.data, p=2, dim=1).T.to(device)

    return alp, det, beta.T


def main_gray():
    """
    Main function to perform SVD on an image and reconstruct it.
    """
    image = Image.open('./figures/Goldhill.png')  # Ensure the image is in RGB format
    
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)

    m, n = image_tensor[0].shape
    final_image = torch.zeros((1, m, n))

    start_time2 = time.time()  

    k = 50
    device = torch.device("cpu")
    
    res = parallel_svd(image_tensor[0], k, device=device)
    if res is None:
        return 
    
    U, S, V = res
    reconstructed_matrix = (U @ S @ V).to(device)

    final_image[0] = reconstructed_matrix

    print("Reconstruction accuracy:", 1 - (final_image - image_tensor).norm() / image_tensor.norm())  # Print reconstruction error
    print(f"Newton SVD takes {time.time() - start_time2:.4f} seconds")


if '__name__' == "__main__":
    main_gray()
