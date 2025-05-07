import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import functional
import matplotlib.pyplot as plt
from PIL import Image
import time
from utils import *

def process_columns_batch(batch_indices, Phi, Psi, W, V, E, R, K, D, N, M, max_iter, gamma_diag, device):
    """
    Process a batch of columns.
    Parameters:
        batch_indices: Indices of the columns to process
        Phi: Transformed matrix
        Psi: Transformed matrix
        W, V, E, R: Matrices to be updated
        K, D, N, M: Dimensions of the matrices
        max_iter: Maximum number of iterations for optimization
        gamma_diag: Diagonal elements of the gamma matrix
        device: Device for computation ("cpu" or "cuda")
    """


    # Start processing a batch of columns
    batch_size = len(batch_indices)
    column_index = batch_indices[0]
    
    # Extract current batch data
    w = W[:, column_index: column_index + batch_size].to(device).reshape(D, batch_size)
    v = V[:, column_index: column_index + batch_size].to(device).reshape(D, batch_size)
    e = E[column_index:column_index + batch_size, :].to(device).reshape(batch_size, N)
    r = R[column_index:column_index + batch_size, :].to(device).reshape(batch_size, M)
    gamma = torch.diag(gamma_diag[column_index:column_index + batch_size]).to(device)
    
    # Construct matrix A and vector b
    A, b = construct_A_and_b(Phi, Psi, D, batch_size, N, M, device)
    
    v_ = torch.randn(A.shape[0]).to(device)
    params = [w, v, e, r]
    
    start_time2 = time.time()
    for iteration in range(max_iter):
        # Define the loss function for the batch
        def loss_func(p, Phi, Psi, gamma, batch_size):
            w, v, e, r = p[:D * batch_size], p[D * batch_size:2 * D * batch_size], p[2 * D * batch_size:2 * D * batch_size + N * batch_size], p[2 * D * batch_size + N * batch_size:2 * D * batch_size + N * batch_size + M * batch_size]
            w, v, e, r = w.view(D, batch_size), v.view(D, batch_size), e.view(batch_size, N), r.view(batch_size, M)
            return (-torch.trace(torch.matmul(w.T, v).to(device)) +
                    0.5 * torch.trace(torch.matmul(torch.matmul(e.T, gamma).to(device), e)) +
                    0.5 * torch.trace(torch.matmul(torch.matmul(r.T, gamma).to(device), r)))

        # Compute gradients and Hessian (first iteration vs others)
        if iteration == 0:
            grad, hessian = compute_gradient_and_hessian(params, loss_func, (Phi, Psi, gamma, batch_size), False, gamma, device)
        else:
            grad, _ = compute_gradient_and_hessian(params, loss_func, (Phi, Psi, gamma, batch_size), True, gamma, device)
        
        # Update parameters using the KKT conditions
        params_tensor = torch.cat([p.reshape(-1) for p in params]).to(device)
        params_tensor.requires_grad_()
        params, v_ = update_params_kkt(params, grad, hessian, A, b, v_, device)

    # Update the matrices with optimized parameters
    W[:, column_index: column_index + batch_size] = w.reshape(-1, batch_size)
    V[:, column_index: column_index + batch_size] = v.reshape(-1, batch_size)
    E[column_index:column_index + batch_size, :] = e.reshape(batch_size, -1)
    R[column_index:column_index + batch_size, :] = r.reshape(batch_size, -1)
    
    print(f"Process the batch {batch_indices} costs: {time.time() - start_time2:.4f} seconds")


def grad_constrained_newton_sequence(matrix, phi, psi, gamma_diag, max_iter, epsilon, device):
    """
    Gradient Constrained Newton method for SVD.

    Parameters:
        matrix: Input matrix of shape (N, M)
        phi: Function to transform rows of the matrix
        psi: Function to transform columns of the matrix
        gamma_diag: Diagonal elements of the gamma matrix
        max_iter: Maximum number of iterations for optimization
        device: Device for computation ("cpu" or "cuda")
    """

    N, M = matrix.shape
    D = phi(matrix[0, :]).shape[1]
    K = len(gamma_diag)
    
    # Create the Gamma matrix as a diagonal matrix from gamma_diag
    Gamma = torch.diag(gamma_diag).to(device)
    
    # Compute Phi and Psi matrices
    Phi = torch.stack([phi(matrix[i, :]) for i in range(N)]).to(device).reshape((N, D))
    Psi = torch.stack([psi(matrix[:, j]) for j in range(M)]).to(device).mT.reshape((M, D))

    # Initialize parameters W, V, E, R with small random values
    W = torch.randn((D, K), requires_grad=False).to(device) * 1e-6
    V = torch.randn((D, K), requires_grad=False).to(device) * 1e-6
    E = torch.randn((K, N), requires_grad=False).to(device) * 1e-6
    R = torch.randn((K, M), requires_grad=False).to(device) * 1e-6

    batch_size = 1
    for column_index in range(0, K, batch_size):
        batch = [column_index + i for i in range(batch_size) if (column_index + i) < K]
        process_columns_batch(batch, Phi, Psi, W, V, E, R, K, D, N, M, max_iter, gamma_diag, device)
    
    # Normalize E and R
    E = torch.pinverse(Gamma) @ torch.nn.functional.normalize(E.data, p=2, dim=1).to(device)
    R = torch.pinverse(Gamma) @ torch.nn.functional.normalize(R.data, p=2, dim=1).to(device)
    
    return [W, V, E, R], 0, Phi, Psi, Gamma


def svd_sequence(matrix, k, n_max, epsilon, device):
    """
    Perform SVD using the Descent method sequentially.
    Parameters:
        matrix: Input matrix of shape (N, M)
        k: Target number of singular values
        n_max: Maximum number of iterations for optimization
        epsilon: Tolerance for convergence
        device: Device for computation ("cpu" or "cuda")
    Returns:
        alp: Left singular vectors
        det: Diagonal singular values
        beta: Right singular vectors
    """
    matrix = matrix.to(device)
    N, M = matrix.shape
    
    # Compute singular values using Lanczos method
    S = lanczos_svd_torch(matrix, k, device=device)

    # Choose compatible matrix using a helper function
    C = choose_compatible_matrix(matrix)
    
    # Define the phi and psi functions for processing
    def phi(x):
        transformed = C.T @ x.float()  # Ensure x is a float
        return transformed.unsqueeze(0) if transformed.dim() == 1 else transformed

    def psi(z):
        return z.unsqueeze(0) if z.dim() == 1 else z

    # Apply gradient-constrained Newton method for optimization
    result, final_cost, Phi, Psi, Gamma = grad_constrained_newton_sequence(matrix, phi, psi, 1/S, n_max, epsilon, device)

    W_opt = result[0]
    V_opt = result[1]
    E_opt = result[2]
    R_opt = result[3]

    # Create expanded diagonal matrix from singular values
    det = create_expanded_diagonal(S, N, M, device)

    # Normalize E and R
    alp = (Gamma @ E_opt).T.to(device)
    beta = (Gamma @ R_opt).T.to(device)

    # Normalize columns of alp and beta
    alp_col_norm = torch.nn.functional.normalize(alp, p=2, dim=0)
    beta_col_norm = torch.nn.functional.normalize(beta, p=2, dim=0)

    alp = alp_col_norm
    beta = beta_col_norm

    # Final matrix approximation
    final_matrix = (alp) @ det @ beta.T
    
    print("Final accuracy", 1 - (final_matrix.data - matrix).norm() / matrix.norm())
    
    return alp, det, beta.T


if __name__ == "__main__":
    # Open and process the image
    image = Image.open('./figures/Goldhill.png')  # Ensure the image is in RGB format
    
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)

    m, n = image_tensor[0].shape
    final_image = torch.zeros((1, m, n))

    # Start processing time
    start_time = time.time()  
    a = 20

    # Perform SVD on the image tensor
    U, S, V = svd_sequence(image_tensor[0], a, device="cpu")

    # Reconstruct the image
    final_image[0] = (U @ S @ V).data

    print(f"SVD takes {time.time() - start_time:.4f} seconds")

    img_numpy = image_tensor.squeeze(0).numpy()
    npimg2 = final_image.data.squeeze(0).numpy()

    # Display original and processed images
    plt.figure(figsize=(10, 5))  # Width 10 inches, height 5 inches

    # First subplot (original image)
    plt.subplot(1, 2, 1)
    plt.imshow(img_numpy, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')  # Turn off axis

    # Second subplot (processed image)
    plt.subplot(1, 2, 2)
    plt.imshow(npimg2, cmap='gray')
    plt.title('Processed Image')
    plt.axis('off')  # Turn off axis

    # Save the images
    plt.savefig('./new_male_smo2.png')
