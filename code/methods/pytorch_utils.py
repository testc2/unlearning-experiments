
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import random 
import os

device = torch.device("cpu")

def lr_loss(w, X, y, lam):
    bce_loss = F.binary_cross_entropy_with_logits(X.mv(w),y.type_as(w),reduction="mean")
    # l2_loss = lam * w.pow(2).sum() * 0.5 #* X.size(0)
    return bce_loss #+ l2_loss

def lr_eval(w, X, y):
    return X.mv(w).sign().eq(y).float().mean()

def lr_grad(w, X, y, lam):
    z = torch.sigmoid(X.mv(w))
    # return (X.t().mv(z -y)/(X.size(0))) + lam * w
    return (X.t().mv(z -y) + lam * X.size(0) * w)

def lr_hessian_inv(w, X, y, lam, batch_size=50000):
    global device
    z = torch.sigmoid(X.mv(w))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i+1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
    # H = (H/X.size(0)) + lam * torch.eye(X.size(1))
    H = (H + lam * X.size(0) * torch.eye(X.size(1)))
    return (H.float().to(device)).inverse()

def lr_optimize_lbfgs(X, y, lam, b=None, num_steps=100, tol=1e-10, verbose=False):
    w = torch.autograd.Variable(torch.zeros(X.size(1)).float().to(device), requires_grad=True)
    def closure():
        if b is None:
            return lr_loss(w, X, y, lam)
        else:
            return lr_loss(w, X, y, lam) + b.dot(w) #/ X.size(0)
    optimizer = optim.LBFGS([w], tolerance_grad=tol, tolerance_change=1e-20)
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) #/ X.size(0)
        loss.backward()
        if verbose and i%10 == 0:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i+1, loss.cpu(), w.grad.norm()))
        optimizer.step(closure)
    return w.data

def lr_optimize_sgd(X,y,params,args,b=None):
    torch.manual_seed(params["sgd_seed"])
    w = torch.autograd.Variable((torch.randn(X.size(1))/math.sqrt(X.size(1))).float().to(device), requires_grad=True)

    if args.optim == "Adam":
        optimizer = optim.Adam([w],lr=args.step_size,weight_decay=args.lam)#,momentum=0.9)
    elif args.optim == "SGD":
        optimizer = optim.SGD([w],lr=args.step_size,weight_decay=args.lam)#,momentum=0.9)
    else:
        raise ValueError("Check optimizer values")
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[200],gamma=0.1)
    for i in range(args.num_steps):
        optimizer.zero_grad()
        loss = lr_loss(w,X,y,args.lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        loss.backward()
        if args.verbose and i%10 == 0:
            print(f"Iteration {i}: lr = {optimizer.param_groups[0]['lr']:.4f} loss = {loss.cpu():.6f}, grad_norm = {w.grad.norm():.6f}")
        optimizer.step()
        if args.lr_schedule:
            scheduler.step()
    return w.data

def lr_optimize_sgd_batch(X,y,params,args,b=None):
    torch.manual_seed(params["sgd_seed"])
    w = torch.autograd.Variable((torch.randn(X.size(1))/math.sqrt(X.size(1))).float().to(device), requires_grad=True)
    if args.optim == "Adam":
        optimizer = optim.Adam([w],lr=args.step_size,weight_decay=args.lam)#,momentum=0.9)
    elif args.optim == "SGD":
        optimizer = optim.SGD([w],lr=args.step_size,weight_decay=args.lam)#,momentum=0.9)
    else:
        raise ValueError("Check optimizer values")
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[500],gamma=0.1)
    n_batches = int(X.size(0)/args.batch_size)
    torch.manual_seed(0)
    for epoch in range(args.num_steps):
        perm = torch.randperm(X.size(0))
        X_train = X.index_select(0,perm)
        y_train = y.index_select(0,perm)
        epoch_loss = []
        for batch in range(0,X.size(0),args.batch_size):
            # print(f"batch={batch}")
            X_batch = X_train[batch:batch+args.batch_size]
            y_batch = y_train[batch:batch+args.batch_size]
            optimizer.zero_grad()
            loss = lr_loss(w,X_batch,y_batch,args.lam)
            if b is not None:
                loss += (b.dot(w) / X_train.size(0))
            loss.backward()
            epoch_loss.append(loss.detach().cpu().item())
            optimizer.step()
        if args.lr_schedule:
            scheduler.step()
        if args.verbose and epoch == args.num_steps-1:
            print(f"Epoch: {epoch}, lr = {optimizer.param_groups[0]['lr']:.4f} Loss: {torch.tensor(epoch_loss).mean()}, grad_norm = {w.grad.norm():.6f}")
    
    return w.data

def onehot(y):
    y_onehot = torch.zeros(y.size(0), y.max() + 1).float()
    y_onehot.scatter_(1, y.long().unsqueeze(1), 1)
    return y_onehot

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def predict_proba(w,X):
    return torch.sigmoid(X.mv(w))

def predict(w,X):
    return predict_proba(w,X).round()

def predict_log_proba(w,X):
    return F.logsigmoid(X.mv(w))
