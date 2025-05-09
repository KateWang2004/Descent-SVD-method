# Descent SVD Method

This is the code repository for our descent SVD method.  

## Requirements

- Python $\geq$ 3.8
- Torch
- Torchvision
- PIL  (for image processing)
- Matplotlib (for visualization）
- mpi4py (for parallelization）



## Code structure

`main.py` is the executive file for our descent SVD method, and `randomized_svd.py` is the classic randomized SVD algorithm used for comparision.

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





