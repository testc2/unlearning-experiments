from types import new_class
from methods.pytorch_utils import lr_hessian_inv,lr_grad
import torch


def batch_remove(w,X_prime,X_remove,y_prime,y_remove,lam):
    H_inv = lr_hessian_inv(w, X_prime, y_prime, lam)
    grad = lr_grad(w,X_remove,y_remove,lam)
    Delta = H_inv.mv(grad)
    w += Delta

    return w

def remove_minibatch_pytorch(w,data,minibatch_size,args,X_remove=None,X_prime=None,y_remove=None,y_prime=None):
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
    for batch in range(0,n_removes,minibatch_size):
        X_batch_remove = X_remove[batch:batch+minibatch_size]
        y_batch_remove = y_remove[batch:batch+minibatch_size]
        X_batch_prime = X_train[batch+minibatch_size:]
        y_batch_prime = y_train[batch+minibatch_size:]
        w_approx = batch_remove(w_approx,X_batch_prime,X_batch_remove,y_batch_prime,y_batch_remove,args.lam)

    return w_approx

def remove_ovr_minibatch_pytorch(w,data,minibatch_size,args,X_remove=None,X_prime=None,y_remove=None,y_prime=None):
    w_approx = w.clone()
    if X_remove is None:
        X_prime = data["X_prime"]
        X_remove = data["X_remove"]
        y_prime = data["y_prime"]
        y_remove = data["y_remove"]
    n_removes = X_remove.size(0)
    n_classes = y_prime.size(1)
    X_train = torch.cat((X_remove,X_prime))
    y_train = torch.cat((y_remove,y_prime))
    for batch in range(0,n_removes,minibatch_size):
        for k in range(n_classes):
            X_batch_remove = X_remove[batch:batch+minibatch_size]
            y_k_batch_remove = y_remove[batch:batch+minibatch_size,k]
            X_batch_prime = X_train[batch+minibatch_size:]
            y_k_batch_prime = y_train[batch+minibatch_size:,k]
            w_approx[:,k] = batch_remove(w_approx[:,k],X_batch_prime,X_batch_remove,y_k_batch_prime,y_k_batch_remove,args.lam)
    
    return w_approx