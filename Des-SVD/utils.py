import torch

def lanczos_svd_torch(matrix, k, device, max_iter=100):
    """
    Compute the top k singular values of a matrix using the Lanczos method.
    Args:
        matrix (torch.Tensor): Input matrix of shape (n, m).
        k (int): Number of singular values to compute.
        device (str): Device to perform computation on ("cpu" or "cuda").
        max_iter (int): Maximum number of Lanczos iterations.
    Returns:
        torch.Tensor: Top k singular values of the matrix.
    """
    # Move the matrix to the specified device
    matrix = matrix.to(device)
    n, m = matrix.shape
    max_iter = min(max_iter, m)  # Prevent max_iter from exceeding the number of columns
    
    # Initialize vectors and Lanczos iteration variables
    v = torch.randn(m, device=device)  # Randomly initialize the initial vector
    v = v / torch.norm(v)  # Normalize the vector
    V = torch.zeros((max_iter, m), device=device)  # Store Lanczos vectors
    alpha = torch.zeros(max_iter, device=device)   # Diagonal elements
    beta = torch.zeros(max_iter, device=device)    # Off-diagonal elements

    # Lanczos iteration
    for j in range(max_iter):
        # Store the current vector
        V[j] = v
        
        # Compute w = A.T @ (A @ v)
        w = matrix.T @ (matrix @ v)
        
        # Compute diagonal element alpha[j]
        alpha[j] = torch.dot(w, v)
        
        # Remove the contribution of the previous vector
        w = w - alpha[j] * v - (beta[j-1] * V[j-1] if j > 0 else 0)
        
        # Re-orthogonalize
        for i in range(j):
            w -= torch.dot(w, V[i]) * V[i]
        
        # Compute the off-diagonal element beta[j]
        beta[j] = torch.norm(w)
        if beta[j] < 1e-10:  # Convergence condition
            break
        
        # Update the vector v
        v = w / beta[j]  # Normalize the vector

    # Construct the tridiagonal matrix T
    T = torch.diag(alpha[:j+1]) + torch.diag(beta[:j], diagonal=1) + torch.diag(beta[:j], diagonal=-1)
    
    # Compute the eigenvalues of the tridiagonal matrix T
    eigvals, _ = torch.linalg.eigh(T)  # Compute eigenvalues and eigenvectors of the tridiagonal matrix
    
    # Take the square root of the eigenvalues as singular values and sort them
    singular_values = torch.sqrt(torch.abs(eigvals))
    singular_values, _ = torch.sort(singular_values, descending=True)
 
    # Return the top k largest singular values
    return singular_values[:k]

def construct_A_and_b(Phi, Psi, D, K, N, M, device):
    """
    Constructs the matrix A and vector b for the KKT conditions.
    Parameters:
        Phi: Transformed matrix
        Psi: Transformed matrix
        D, K, N, M: Number of features
        device: Device for computation ("cpu" or "cuda")
    Returns:
        A: Matrix A for KKT conditions
        b: Vector b for KKT conditions
    """
    # Length of parameter vectors
    num_w = D * K
    num_v = D * K
    num_e = K * N
    num_r = K * M
  
    num_params = num_w + num_v + num_e + num_r
    
    # Construct matrix A
    A = torch.zeros((num_e + num_r, num_params)).to(device)

    # Constraint for E = (Phi W)^T
    for n in range(N):
        for k in range(K):
            # Since W is stored by rows, for E(k, n), it is affected by all D rows of W
            A[k * N + n, k:D*K:K] = Phi[n, :]  # Adjust index mapping to match flattened W
            A[k * N + n, num_w + num_v + k * N + n] = -1  # E is at this position
    
    # Constraint for R = (Psi V)^T
    for m in range(M):
        for k in range(K):
            A[num_e + k * M + m, (num_w + k):(num_w + D * K):K] = Psi[m, :]  # Adjust index mapping for flattened V
            A[num_e + k * M + m, num_w + num_v + num_e + k * M + m] = -1  # R is at this position

    # Vector b is a zero vector, because the constraints should be fully satisfied
    b = torch.zeros(num_e + num_r)

    return A, b


def choose_compatible_matrix(matrix):
    """
    Choose the compatible matrix for pseudo-inverse calculation.
    Args:
        matrix (torch.Tensor): Input matrix.
    Returns:
        torch.Tensor: The compatible matrix
    """
    N, M = matrix.shape
    if N < M:
        return (torch.pinverse(matrix @ matrix.T) @ matrix).T
    elif N > M:
        return torch.pinverse(matrix.T @ matrix) @ matrix.T
    elif N == M:
        return torch.pinverse(matrix)

def create_expanded_diagonal(S, m, n, device):
    """
     Create an expanded diagonal matrix from the singular values
        Args:
            S (torch.Tensor): Singular values.
            m (int): Number of rows in the original matrix.
            n (int): Number of columns in the original matrix.
            device (str): Device for computation ("cpu" or "cuda").
        Returns:
            torch.Tensor: Expanded diagonal matrix.
    """
    min_dim = min(m, n, len(S))
    expanded_S = torch.zeros((min_dim, min_dim)).to(device)
    for i in range(min_dim):
        expanded_S[i, i] = S[i]
    return expanded_S


def compute_gradient_and_hessian(params, loss_func, args, calulated, gamma, device):
    """
    Compute the gradient and Hessian of the loss function with respect to the parameters.
    Parameters:
        params: List of parameters (W, V, E, R)
        loss_func: Loss function to compute the loss
        args: Additional arguments for the loss function
        calulated: Boolean indicating whether to calculate Hessian
        gamma: 1/s
        device: Device for computation ("cpu" or "cuda")
    Returns:
        grad: Gradient of the loss function
        hessian: Hessian of the loss function
    """

    params_tensor = torch.cat([p.reshape(-1) for p in params]).to(device)
    params_tensor.requires_grad_()
    
    # Compute loss and gradient
    loss = loss_func(params_tensor, *args)
    
    # Compute gradient
    grad = torch.autograd.grad(loss, params_tensor)[0]

    
    # Compute Hessian using autograd.functional.hessian
    if calulated:
        hessian = None
    else:
        hessian = calculate_hessian(params, gamma, device)
        
    # Return gradient and Hessian
    return grad, hessian

def calculate_hessian(params, gamma, device):
    """
    Calculate the Hessian matrix for the parameters.
    Parameters:
        params: List of parameters (W, V, E, R)
        gamma: 1/s
        device: Device for computation ("cpu" or "cuda")
    Returns:
        hessian: Hessian matrix
    """


    # Flatten and concatenate parameters to ensure they are on the correct device
    params_tensor = torch.cat([p.reshape(-1) for p in params]).to(device)
    w_size = params[0].numel()
    v_size = params[1].numel()
    e_size = params[2].numel()
    r_size = params[3].numel()
    
    # Initialize Hessian matrix
    hessian = torch.zeros((w_size + v_size + e_size + r_size, w_size + v_size + e_size + r_size)).to(device)

    # Fill the Hessian matrix
    for i in range(w_size + v_size + e_size + r_size):
        if i < w_size:
                hessian[i,i+w_size] = -1.
                hessian[i+w_size,i] = -1.
        elif i >=(w_size+v_size) and i < (w_size+v_size+e_size):
            hessian[i,i] = gamma
        elif i >= (w_size+v_size+e_size) and i < (w_size+v_size+e_size+r_size):
            hessian[i,i] = gamma
                
    # Return the Hessian matrix
    return hessian  


def update_params_kkt(params, grad, hessian, A, b, v, device, alpha=0.1, beta=0.7, epsilon0=1e-4, max_iter=50):

    """
    Update parameters using the KKT conditions.
    Parameters:
        params: List of parameters (W, V, E, R)
        grad: Gradient of the loss function
        hessian: Hessian of the loss function
        A: Matrix A for KKT conditions
        b: Vector b for KKT conditions
        v: Dual variable
        device: Device for computation ("cpu" or "cuda")
        alpha: Step size parameter
        beta: Step size reduction factor
        epsilon0: Convergence threshold
        max_iter: Maximum number of iterations for KKT update
    Returns:
        params: Updated parameters
        v: Updated dual variable
    """
    # Automatically select the device
  
    num_params = sum(p.numel() for p in params)
    total_vars = num_params + A.shape[0]

    # Move all input tensors to the GPU
    A = A.to(device)
    b = b.to(device)
    grad = grad.to(device)
    hessian = hessian.to(device)
    v = v.to(device)
    
    # Create KKT matrix and vector
    KKT_matrix = torch.zeros((total_vars, total_vars), device=device)
    KKT_vector = torch.zeros(total_vars, device=device)
    
    # Initialize KKT matrix
    KKT_matrix[:num_params, :num_params] = hessian
    KKT_matrix[:num_params, num_params:] = A.T
    KKT_matrix[num_params:, :num_params] = A
    
    # Collect the current parameters into a vector
    x = torch.cat([p.data.reshape(-1) for p in params]).to(device)
    
    # Residual r
    r = A @ x - b

    for iteration in range(max_iter):
        # Update KKT vector
        KKT_vector[:num_params] = -grad - A.T @ v
        KKT_vector[num_params:] = b - A @ x
        
        # Solve KKT system
        delta = torch.linalg.solve(KKT_matrix, KKT_vector)
        
        # Split delta into updates for x and v
        delta_x_nt = delta[:num_params]
        delta_v_nt = delta[num_params:]
        
        # Line search for optimal step size
        t = 1
        while t > 1e-10:
            new_x = x + t * delta_x_nt
            new_v = v + t * delta_v_nt
            new_r = A @ new_x - b
            if torch.norm(new_r) <= (1 - alpha * t) * torch.norm(r):
                break
            t *= beta
        
        # Update parameters
        x += t * delta_x_nt
        v += t * delta_v_nt
        
        # Update params object
        index = 0
        for p in params:
            numel = p.numel()
            p.data = x[index:index + numel].reshape(p.shape)
            index += numel
        
        # Check convergence condition
        if torch.norm(new_r) < epsilon0:
            break
        
        r = new_r  # Update residual
    
    return params, v
