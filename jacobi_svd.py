import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import functional
import matplotlib.pyplot as plt
from PIL import Image
import time


def randomized_svd(A, k=100, n_iter=10, device="cpu"):
    """
    Randomized SVD method

    Parameters:
      A: Input matrix, shape (m, n)
      k: Target rank, compute the first k singular values/vectors
      n_iter: Number of power iterations to improve the approximation accuracy (default is 1, can increase to 2-3 iterations)
      device: Computation device ("cpu" or "cuda")

    Returns:
      U: Approximate left singular vectors, shape (m, k)
      S: Approximate singular value diagonal matrix, shape (k, k)
      VT: Approximate transpose of the right singular vectors, shape (k, n)
    """
    # Move A to the target device
    A = A.to(device)
    m, n = A.shape
    time1 = time.time()

    # 1. Generate random projection matrix Omega (n x k)
    Omega = torch.randn(n, k, device=device)
    # 2. Compute Y = A * Omega (m x k)
    Y = A @ Omega
    # 3. Perform QR decomposition on Y to get orthogonal matrix Q (m x k)
    Q, _ = torch.linalg.qr(Y, mode="reduced")

    # Optionally, perform several power iterations to improve approximation accuracy
    for _ in range(n_iter):
        Y = A.T @ Q  # (n x k)
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        Y = A @ Q    # (m x k)
        Q, _ = torch.linalg.qr(Y, mode="reduced")
    
    time3 = time.time()
    numpy_array = A.cpu().numpy()

    # 4. Construct a small matrix B = Qᵀ * A, (k x n) in the Q subspace
    B = Q.T @ A
    
    # 5. Perform SVD on B: B = U_B * S * VT, where U_B (k x k), S (k,), VT (k x n)
    U_B, S_vec, VT = qr_jacobi_svd(B.T, device=device)

    U_B_ = VT.T
    VT = U_B.T
    U_B = U_B_

    # 6. Compute U = Q * U_B, shape (m x k)
    U = Q @ U_B

    # Convert singular value vector to diagonal matrix
    S = S_vec

    return U, S, VT


def jacobi_svd(X, eps=1e-6, maxiter=100, device='cpu'):
    """
    Compute SVD using Jacobi method, supports CPU and CUDA.
    Assume X is an m×n matrix (m >= n), return reduced SVD:
      X ≈ U * S * VT, where U is m×n, S is n×n diagonal matrix, VT is n×n.
    Parameters:
        X: Input matrix (m, n)
        eps: Convergence threshold
        maxiter: Maximum number of iterations
        device: Computation device ("cpu" or "cuda")
    Returns:
        U: m×n left singular vector matrix
        S: n×n diagonal singular value matrix
        VT: n×n right singular vector transpose
    """

    X = X.to(device)
    m, n = X.shape

    V = torch.eye(n, device=device, dtype=X.dtype)
    X = X.clone()

    start_time = time.time()

    for iter_num in range(maxiter):
        converged = True
        for p in range(n):
            for q in range(p + 1, n):
                x_p = X[:, p]
                x_q = X[:, q]
                cross = torch.dot(x_p, x_q)
                norm_p = torch.norm(x_p, p=2)
                norm_q = torch.norm(x_q, p=2)

                if abs(cross) > eps * norm_p * norm_q:
                    converged = False

                    # Compute Jacobi rotation
                    tau = (norm_q**2 - norm_p**2) / (2 * cross)
                    t = torch.sign(tau) / (abs(tau) + torch.sqrt(1 + tau**2))
                    c = 1 / torch.sqrt(1 + t**2)
                    s = c * t

                    J = torch.eye(n, device=device, dtype=X.dtype)
                    J[p, p] = c
                    J[q, q] = c
                    J[p, q] = s
                    J[q, p] = -s

                    X = X @ J
                    V = V @ J

        if converged:
            break

    Sigma = torch.norm(X, dim=0)
    U = X / Sigma.clamp(min=eps)  # Avoid division by zero

    sorted_indices = torch.argsort(Sigma, descending=True)
    Sigma = Sigma[sorted_indices]
    U = U[:, sorted_indices]
    V = V[:, sorted_indices]

    S = torch.diag(Sigma)

    return U, S, V.T


def qr_jacobi_svd(A, max_iter=100, tol=1e-4, device="cpu"):
    """
    Compute SVD using QR-Jacobi method, supports CPU and CUDA.
    Assume A is an m×n matrix (m >= n), return reduced SVD:
      A ≈ U * S * VT, where U is m×n, S is n×n diagonal matrix, VT is n×n.

    Parameters:
      A: Input matrix (m, n)
      max_iter, tol: Parameters passed to Jacobi iteration for SVD computation of R
      device: Computation device ("cpu" or "cuda")

    Returns:
      U_final: m×n left singular vector matrix
      S_matrix: n×n diagonal singular value matrix
      VT: n×n right singular vector transpose
    """
    A = A.to(device)
    m, n = A.shape

    # Step 1: QR decomposition, A = Q * R
    Q, R = torch.linalg.qr(A, mode="reduced")  # Q: m×n, R: n×n

    # Step 2: Perform Jacobi SVD on R to get R = U_r * S * VT
    U_r, S_matrix, VT = jacobi_svd(R)

    # Step 3: Final left singular vectors U = Q * U_r
    U_final = Q @ U_r  # m×n

    return U_final, S_matrix, VT


def main_gray(path,k,device,visualize=False):

    image = Image.open(path)  # Ensure the image is in RGB format
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.CenterCrop(50)
        #  transforms.RandomCrop(505)
    ])
    print("Image size:", image.size)

    # Apply transformations
    image_tensor = transform(image)
    
    image_tensor = image_tensor[0].unsqueeze(0)

    m, n = image_tensor[0].shape
    final_image = torch.zeros((1, m, n))

    start_time2 = time.time()
    U, S, V = randomized_svd(image_tensor[0], k, device=device)

    reconstructed_matrix = (U @ S @ V).to(device)

    image_tensor = image_tensor.to(device)

    final_image[0] = reconstructed_matrix

    print("Reconstruction accuracy:", 1. -  (final_image - image_tensor).norm() / image_tensor.norm())  # Output reconstruction error
    print(f"Classic Randomized svd takes {time.time() - start_time2:.4f} seconds")

    if visualize:
        plot_image(final_image, './figures/cat_reconstructed.jpg')


def plot_image(final_image,output_path):
    """
    Plot the final image and save it to the specified path.
    Parameters:
    final_image (tensor): The final image tensor.
    output_path (str): Path to save the image.
    """
    transform = transforms.ToPILImage()
    pil_image = transform(final_image)
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(pil_image,cmap='gray')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    # Set the device to CUDA if available, otherwise use CPU
    device = torch.device("cpu") # Recommended
    print(f"Using device: {device}")

    # Example usage
    path = "./figures/cat.jpg"  # Replace with your image path
    k = 100  # Target rank
    main_gray(path, k, device, visualize=True)
