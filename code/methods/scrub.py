import torch
from methods.pytorch_utils import lr_hessian_inv,lr_grad
from scipy.linalg import fractional_matrix_power

def compute_noise(w,X_prime,y_prime,lam,noise,noise_seed,B_inv=None):
    """Computes the noise for Fisher noise.

    Args:
        w (torch.tensor): The model weights
        X_prime (torch.tensor): The remaining training samples
        y_prime (torch.tensor): The labels of the remaining samples
        lam (float): The regularization term
        noise (float): The quantity of noise to add to the model
        noise_seed (int): The seed for the random noise
        B_inv (torch.tensor, optional): The Hessian inverse if already computed. Defaults to None.
    """
    if noise > 0:
        torch.manual_seed(noise_seed)
        # compute hessian inverse if not avaible
        if B_inv is None:
            B_inv = lr_hessian_inv(w,X_prime,y_prime,lam)
        
        gauss_noise = torch.randn_like(w)
        # cov = torch.from_numpy(fractional_matrix_power(B_inv,1/2).real).float()
        # cov_chol  = torch.cholesky(cov)
        # noise_scrub = noise*cov_chol.mv(gauss_noise)
        B_inv_frac = torch.from_numpy(fractional_matrix_power(B_inv,1/4).real).float()
        noise_scrub = noise*B_inv_frac.mv(gauss_noise)
    else:
        noise_scrub = torch.zeros_like(w)

    return noise_scrub


def scrub(w,X_prime,y_prime,lam,noise,noise_seed):
    B_inv = lr_hessian_inv(w,X_prime,y_prime,lam)
    grad = lr_grad(w,X_prime,y_prime,lam)
    quadratic_scrub = B_inv.mv(grad)
    # get the noise to be added for the method
    noise_scrub = compute_noise(w,X_prime,y_prime,lam,noise,noise_seed,B_inv=B_inv)
    w += (noise_scrub - quadratic_scrub)

    return w, noise_scrub

def scrub_minibatch_pytorch(w,data,minibatch_size,args,noise,noise_seed,X_remove=None,X_prime=None,y_remove=None,y_prime=None):
    w_approx = w.clone()
    if X_remove is None:    
        X_prime = data["X_prime"]
        X_remove = data["X_remove"]
        y_prime = data["y_prime"]
        y_remove = data["y_remove"]
    n_removes = X_remove.size(0)
    # concatenate the remove and remaining 
    X_train = torch.cat((X_remove,X_prime))
    y_train = torch.cat((y_remove,y_prime))
    total_added_noise = torch.zeros_like(w)
    for batch in range(0,n_removes,minibatch_size):
        X_batch_prime = X_train[batch+minibatch_size:]
        y_batch_prime = y_train[batch+minibatch_size:]
        w_approx,added_noise = scrub(w_approx,X_batch_prime,y_batch_prime,args.lam,noise,noise_seed)
        total_added_noise += added_noise
    return w_approx,total_added_noise

def scrub_ovr_minibatch_pytorch(w,data,minibatch_size,args,noise,noise_seed,X_remove=None,X_prime=None,y_remove=None,y_prime=None):
    w_approx = w.clone()
    if X_remove is None:
        X_prime = data["X_prime"]
        X_remove = data["X_remove"]
        y_prime = data["y_prime"]
        y_remove = data["y_remove"]
    n_removes = X_remove.size(0)
    n_classes = y_prime.size(1)
    # concatenate the remove and remaining 
    X_train = torch.cat((X_remove,X_prime))
    y_train = torch.cat((y_remove,y_prime))
    total_added_noise = torch.zeros_like(w)    
    for batch in range(0,n_removes,minibatch_size):
        X_batch_prime = X_train[batch+minibatch_size:]
        for k in range(n_classes):
            y_k_batch_prime = y_train[batch+minibatch_size:,k]
            w_temp,added_noise = scrub(w_approx[:,k],X_batch_prime,y_k_batch_prime,args.lam,noise,noise_seed)
            w_approx[:,k] = w_temp
            total_added_noise[:,k] = added_noise
        
    return w_approx,total_added_noise
