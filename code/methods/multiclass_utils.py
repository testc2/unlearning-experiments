from methods.pytorch_utils import lr_grad,lr_loss,lr_hessian_inv, lr_optimize_sgd, lr_optimize_sgd_batch
import torch 
import torch.multiprocessing as mp
from functools import partial


def lr_ovr_optimize_sgd(X_train,y_train,params,args,b=None):
    w = torch.zeros((X_train.size(1),y_train.size(1))).float()
    for k in range(y_train.size(1)):
        if args.verbose:
            print(f"Class: {k} ",end="")
        if b is not None:
            w[:,k] = lr_optimize_sgd_batch(X_train,y_train[:,k],params,args,b[:,k])
        else:
            w[:,k] = lr_optimize_sgd_batch(X_train,y_train[:,k],params,args,None)
    return w

def predict_ovr(w,X):
    return X.mm(w).argmax(1)

def predict_proba_ovr(w,X):
    return torch.softmax(X.mm(w),dim=1)

def predict_log_proba_ovr(w,X):
    return torch.log_softmax(X.mm(w),dim=1)
