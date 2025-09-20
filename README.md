# Descent SVD Method

This is the code repository for our descent SVD method.  

## Requirements

- Python $\geq$ 3.8
- Torch
- Torchvision
- PIL  (for image processing)
- Matplotlib (for visualization)
- mpi4py (for parallel processing; ensure that your device has enough CPU cores available).

## Code structure

`main.py` is the executive file for our descent-based Singular Value Decomposition (SVD) method, which is designed to efficiently compute the singular value decomposition of large matrices using optimization techniques on Riemannian manifolds. This method focuses on improving the speed and robustness of SVD computations, especially for high-dimensional data, through iterative descent-based optimization techniques.

`jacobi_svd.py `implements the classic randomized SVD algorithm. The randomized approach utilizes a combination of random projections and orthogonalization to compute a low-rank approximation of the singular vectors, providing a fast, probabilistic alternative to traditional SVD.

`Riemannian_svd.py` focuses on the implementation of SVD optimization directly on the Stiefel manifold, leveraging the Riemannian geometry of the problem to ensure that the computed singular vectors remain orthogonal throughout the iterations. This method is designed to enhance the stability and convergence speed of the SVD computations under non-Euclidean constraints.

`lanczos_svd.py` implements the Lanczos algorithm, a classical iterative method for computing eigenvalues and eigenvectors. It is commonly used for approximating the singular values and vectors of large matrices. 

The three methods mentioned above—randomized SVD, Riemannian SVD, and Lanczos SVD—are compared with our descent-based SVD (Des-SVD) method in this paper. The comparison aims to highlight the advantages and limitations of each approach in terms of computational efficiency, accuracy, and scalability, especially when dealing with large-scale and high-dimensional data. By benchmarking these methods under various conditions, we demonstrate the effectiveness of our Des-SVD method in terms of both speed and robustness, while showcasing its ability to efficiently handle large, sparse, and noisy matrices.

## Execute SVD

You can select parameters in `main.py`  like:

```python
  # Select the parameters
  image = Image.open('./figures/Hill.png')  # The target image path
  k = 50 # The target number of singular values
  device = torch.device("cpu") # or "cuda" for GPU
  n_max = 5 # The maximum number of iterations for Descent SVD
  epsilon = 1e-10 # The tolerance for convergence
  parallel = True # Whether to use parallel computation
  randomized = True # Whether to use randomized SVD
  visualized = True # Whether to visualize the result
```

Importantly, if you choose to execute the Descent SVD **in parallel** (which is recommended), you should run the following command in the terminal:

```terminal
mpiexec -n k python main.py
```

where the process number is equivalent to `k`.





