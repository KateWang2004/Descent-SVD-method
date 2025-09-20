import torch
import time
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from mpi4py import MPI 

from sequence_svd import svd_sequence

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def randomized_svd_sequential(A, execute, n_max, epsilon, k=50, n_iter=20, device="cpu"):
    """
    Randomized SVD method
    
    Parameters:
      A: Input matrix, shape (m, n)
      execute: Function to execute the SVD computation
      n_max: Maximum number of iterations for the SVD computation
      epsilon: Tolerance for convergence
      k: Target rank, calculate the first k singular values/vectors
      n_iter: Power iteration count, used to improve the approximation accuracy (default is 1, can be increased to 2-3 iterations)
      device: Computation device ("cpu" or "cuda")
      
    Returns:
      U: Approximate left singular vectors, shape (m, k)
      S: Approximate singular value diagonal matrix, shape (k, k)
      VT: Approximate transpose of right singular vectors, shape (k, n)
    """
    # Move A to the target device
    A = A.to(device)
    m, n = A.shape
    
    # 1. Generate random projection matrix Omega (n x k)
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

    # 4. Construct small matrix B = Qᵀ * A, (k x n)
    B = Q.T @ A

    # Perform sequential QR-Jacobi SVD on B
    res = qr_jacobi_svd_sequential(B.T, execute, n_max, epsilon, device=device)
    if res is None:
        return None
    
    U_B, S_vec, VT = res
    U_B_ = VT.T
    VT = U_B.T
    U_B = U_B_
    # 6. Compute U = Q * U_B, shape (m x k)
    U = Q @ U_B
    S = S_vec
    
    return U, S, VT


def qr_jacobi_svd_sequential(A, execute, n_max, epsilon, max_iter=1000, tol=1e-6, device="cpu"):
    """
    QR-Jacobi method for SVD computation, supports CPU and CUDA.
    
    Parameters:
      A: Input matrix (m, n)
      n_max: Maximum number of iterations for the SVD computation
      epsilon: Tolerance for convergence for SVD
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
    Q, R = torch.linalg.qr(A, mode="reduced")  # Q: m×n, R: n×n

    time2 = time.time()
    res = execute(R, R.shape[0], n_max, epsilon, device=device)

    U_r, S_matrix, VT = res
    # Step 3: Final left singular vector U = Q * U_r
    U_final = Q @ U_r  # m×n
    
    return U_final, S_matrix, VT


if __name__ == "__main__":

    image = Image.open('./figures/cat.jpg')  # Ensure the image is in RGB format
    
    transform = transforms.Compose([
            transforms.ToTensor(), 
            # transforms.CenterCrop(50)
        #  transforms.RandomCrop(505)
        ])

    # Apply the transformation
    image_tensor = transform(image)

    m, n = image_tensor[0].shape
    final_image = torch.zeros((1, m, n))

    start_time2 = time.time()  

    k = 100
    device = torch.device("cpu")
    res = randomized_svd_sequential(image_tensor[0], svd_sequence, n_max=3, epsilon=1e-10, k=k, device=device)

    U, S, V = res

    reconstructed_matrix = (U @ S @ V).to(device)

    final_image[0] = reconstructed_matrix

    # If rank == 0:
    print("Reconstruction accuracy:", 1 - (final_image - image_tensor).norm()/image_tensor.norm())  # Print reconstruction error
    print(f"Newton SVD takes {time.time() - start_time2:.4f} seconds")
