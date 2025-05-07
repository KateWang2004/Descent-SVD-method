from PIL import Image
import torch
import time
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
 
from random_parallel import parallel_svd, randomized_svd
from random_sequence import randomized_svd_sequential
from sequence_svd import svd_sequence


def descent_svd(A,k,parallel,randomized,n_max=3,epsilon=1e-10,device="cpu"):
    """
    Perform SVD using the descent method.

    Parameters:
    A (ndarray): Input matrix of shape (m, n).
    k (int): Target rank for SVD.
    parallel (bool): Whether to use parallel computation.
    randomized (bool): Whether to use randomized SVD.
    n_max (int): Maximum number of iterations for randomized SVD.
    epsilon (float): Tolerance for convergence.
    device (str): Device to perform computation on ("cpu" or "cuda").
    
    Returns:
    U (ndarray): Left singular vectors.
    S (ndarray): Singular values.
    V (ndarray): Right singular vectors.
    """
    if(parallel and randomized):
        return randomized_svd(A, k, n_max, epsilon, device=device)
    
    elif(parallel and not randomized):
        return parallel_svd(A, k, n_max, epsilon, device=device)
    
    elif (not parallel and randomized):
        return randomized_svd_sequential(A,svd_sequence, n_max, epsilon, k,device=device)

    else:
        return svd_sequence(A, k, n_max, epsilon, device=device)
        # if(res is None):
        #     return None
        # U, S, V = res
        # print("randomized svd done")




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



def execute(image,k,device,n_max,epsilon,parallel,randomized,visualized):
    """
    Execute the SVD process and plot the image (optional).
    """



    transform = transforms.Compose([
            transforms.ToTensor(), 
        ])

    image_tensor = transform(image)


    m,n = image_tensor[0].shape
    final_image = torch.zeros((1,m,n))

    start_time2 = time.time()  

    res = descent_svd(image_tensor[0],k,parallel,randomized,n_max,epsilon,device=device)

    if res is None:
        return 
    
    U, S, V = res

    reconstructed_matrix = (U @ S @ V).to(device)

    final_image[0] = reconstructed_matrix

    print("Reconstruction accuracy:", 1 - (final_image - image_tensor).norm()/image_tensor.norm())  # 输出重构误差
    print(f"Descent svd takes {time.time() - start_time2:.4f} seconds")

    if visualized:
    # Optionally plot the image
        plot_image(final_image, './figures/gold_reconstructed.jpg')



if __name__ == "__main__":
    # Select the parameters
    image = Image.open('./figures/Goldhill.png')  # The target image
    k = 50 # The target number of singular values
    device = torch.device("cpu") # or "cuda" for GPU
    n_max = 5 # The maximum number of iterations for Descent SVD
    epsilon = 1e-10 # The tolerance for convergence
    parallel = True # Whether to use parallel computation
    randomized = True # Whether to use randomized SVD
    visualized = True # Whether to visualize the result

    # If parallel, use the command in the terminal: mpiexec -n k python main.py
    execute(image,k,device,n_max,epsilon,parallel,randomized,visualized)