# Descent SVD Method

This is the code repository for our descent SVD method. 



## Requirements

- Python $\geq$ 3.8
- Torch
- Torchvision
- PIL
- Matplotlib
- mpi4py



## Execute SVD

You can select parameters in `main.py`  like:

```python
  # Select the parameters
  image = Image.open('./figures/Goldhill.png')  # The target image path
  k = 50 # The target number of singular values
  device = torch.device("cpu") # or "cuda" for GPU
  n_max = 5 # The maximum number of iterations for Descent SVD
  epsilon = 1e-10 # The tolerance for convergence
  parallel = True # Whether to use parallel computation
  randomized = True # Whether to use randomized SVD
  visualized = True # Whether to visualize the result
```

If you select to execute SVD in parallel  (Recommended), you should run in the terminal:

```terminal
mpiexec -n k(the concrete number,like 50) python main.py
```

where the process number is equivalent to `k`.

