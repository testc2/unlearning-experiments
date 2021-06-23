from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils import data
import torchvision
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_svmlight_file

from sklearn.metrics import roc_auc_score,f1_score

def load_mnist(data_dir,ovr=False,chosen_classes=[3,8],l2_norm=True):
    if not (data_dir/"MNIST"/"processed"/"training.pt").exists():
        train = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform= torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,),(0.3081))
        ])
        )
    X_train,y_train = torch.load(data_dir/"MNIST"/"processed"/"training.pt")
    X_test,y_test = torch.load(data_dir/"MNIST"/"processed"/"test.pt")
    X_train = X_train.float().reshape(X_train.size(0),-1)
    X_test = X_test.float().reshape(X_test.size(0),-1)
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train-mean)/std
    X_test = (X_test-mean)/std
    if l2_norm:    
        # L2 normalize the data
        # X_train /= X_train.norm(p=2,dim=1).unsqueeze(1)
        # X_test /= X_test.norm(p=2,dim=1).unsqueeze(1)
        # Alternate L2 norm with the max norm of the dataset
        X_train /= X_train.norm(p=2,dim=1).max()
        X_test /= X_test.norm(p=2,dim=1).max()
    
    if not ovr : 
        train_mask = np.isin(y_train,chosen_classes)
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]

        test_mask = np.isin(y_test,chosen_classes)
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        
        # convert classes 3 and 8 to 0 and 1
        binarizer = LabelBinarizer()
        binarizer.fit(y_train)
        y_train = torch.from_numpy(binarizer.transform(y_train).flatten())
        y_test = torch.from_numpy(binarizer.transform(y_test).flatten())
    else:
        y_train = F.one_hot(y_train)
        y_test = F.one_hot(y_test)
    
    return X_train,X_test,y_train,y_test

def get_remove_prime_splits(X,y,n_removes,remove_type="random",seed=0,cost_indices=None,remove_class=0):
    remove_indices = keep_indices = None
    if remove_type == "high_cost":
        # remove first n_removes indices with highest cost
        remove_indices = cost_indices[:n_removes]
        keep_indices = torch.LongTensor(list(set(range(X.shape[0]))-set(remove_indices.numpy())))
    elif remove_type == "class_high_cost":
        # remove first n_remove indices from cost_class with highest cost
        y_train_sorted = y[cost_indices]
        cost_indices_0 = cost_indices[y_train_sorted==remove_class]
        remove_indices = cost_indices_0[:n_removes]
        keep_indices = torch.LongTensor(list(set(range(X.shape[0]))-set(remove_indices.numpy())))
    elif remove_type == "class":
        y_class_index = torch.arange(X.size(0))[y.argmax(1)==remove_class]
        remove_indices = y_class_index[:n_removes]
        keep_indices = torch.LongTensor(list(set(range(X.size(0)))-set(remove_indices.numpy())))
    elif remove_type == "random":
        # pick random n_remove indices for removal
        torch.manual_seed(seed)
        perm = torch.randperm(X.size(0))
        remove_indices = perm[:n_removes]
        keep_indices = perm[n_removes:]
    else:
        raise ValueError(f"Removal method option incorrect")
    X_remove = X.index_select(0, remove_indices)
    y_remove = y.index_select(0,remove_indices)
    X_prime = X.index_select(0,keep_indices)
    y_prime = y.index_select(0,keep_indices)

    return X_remove,X_prime,y_remove,y_prime    

def create_toy_dataset(l2_norm=True):
    X, y = make_blobs(
        n_samples=2000, centers=[[2, 1], [-1.5, -1]], n_features=2, random_state=41, cluster_std =0.5
    )
    X_remove, y_remove = make_blobs(
        n_samples=100, centers=[[-2, 1]], n_features=2, random_state=42, cluster_std=0.7
    )

    if l2_norm:
        X /= np.linalg.norm(X,ord=2,axis=1)[:,None]
        X_remove /= np.linalg.norm(X_remove,ord=2,axis=1)[:,None]

    X_prime,X_test,y_prime,y_test = train_test_split(X,y,shuffle=False)
    X_train = np.r_[X_prime, X_remove]
    y_train = np.r_[y_prime, y_remove]

    return [torch.torch.from_numpy(a).float() for a in [X_train, X_test, X_prime, X_remove, y_train, y_test, y_prime, y_remove]]

def load_rcv1(data_dir:Path):
    processed_dir = data_dir/"RCV1"/"processed"
    train_file = processed_dir/"training.pt"
    test_file = processed_dir/"test.pt"

    if not train_file.exists() or not test_file.exists():
        processed_dir.mkdir(parents=True,exist_ok=True)
        svm_train = data_dir/"RCV1"/"rcv1_train.binary"
        svm_test = data_dir/"RCV1"/"rcv1_test.binary"
        X_train,y_train = load_svmlight_file(str(svm_train))
        X_train = X_train.tocoo()
        y_train = (y_train+1)/2
        X_train = torch.sparse.FloatTensor(
            torch.LongTensor([X_train.row.tolist(),X_train.col.tolist()]),
            torch.FloatTensor(X_train.data.astype(np.float32))
        )
        y_train = torch.from_numpy(y_train).int()
        print("Storing Files")  
        torch.save([X_train,y_train],f=processed_dir/"training.pt")
        X_test,y_test = load_svmlight_file(str(svm_test))
        _,X_test,_,y_test = train_test_split(X_test,y_test,shuffle=False,test_size=0.05)
        X_test = X_test.tocoo()
        y_test = (y_test+1)/2
        X_test = torch.sparse.FloatTensor(
            torch.LongTensor([X_test.row.tolist(),X_test.col.tolist()]),
            torch.FloatTensor(X_test.data.astype(np.float32))
        )
        y_test = torch.from_numpy(y_test).int()
        print("Storing Files")
        torch.save([X_test,y_test],f=processed_dir/"test.pt")
        return X_train, X_test, y_train, y_test

    
    return X_train,X_test,y_train,y_test

def load_covtype(data_dir,l2_norm=True,ovr=True):
    if ovr:
        return load_covtype_ovr(data_dir,l2_norm)
    else:
        return load_covtype_binary(data_dir,l2_norm)
    
def load_covtype_ovr(data_dir,l2_norm=True):
    processed_dir = data_dir/"COVTYPE"/"processed"
    train_file = processed_dir/"training.pt"
    test_file = processed_dir/"test.pt"

    if not test_file.exists() or not train_file.exists():
        processed_dir.mkdir(parents=True,exist_ok=True)
        Xy = np.genfromtxt(data_dir/"COVTYPE"/"covtype.data",delimiter=",")
        X = Xy[:,:-1]
        y =Xy[:,-1]
        # Change labels from 1..7 to 0..6
        y-=1
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0,stratify=y)
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).long()

        torch.save([X_train,y_train],f=train_file)
        torch.save([X_test,y_test],f=test_file)
    
    else: 
        X_train,y_train = torch.load(train_file)
        X_test,y_test = torch.load(test_file)
    
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train-mean)/std
    X_test = (X_test-mean)/std
    y_train = F.one_hot(y_train)
    y_test = F.one_hot(y_test)
    if l2_norm:    
        # L2 normalize the data
        X_train /= X_train.norm(p=2,dim=1).max()
        X_test /= X_test.norm(p=2,dim=1).max()
    
    X_train = torch.cat((X_train,torch.ones(X_train.size(0),1)),dim=1)
    X_test = torch.cat((X_test,torch.ones(X_test.size(0),1)),dim=1)
    return X_train, X_test, y_train,y_test

def load_covtype_binary(data_dir,l2_norm=True):
    processed_dir = data_dir/"COVTYPE"/"processed"
    train_file = processed_dir/"binary_training.pt"
    test_file = processed_dir/"binary_test.pt"

    if not test_file.exists() or not train_file.exists():
        processed_dir.mkdir(parents=True,exist_ok=True)
        X,y = load_svmlight_file(str((data_dir/"COVTYPE"/"covtype.libsvm.binary.scale").resolve()))
        # Change labels from 1,2 to 0,1
        y-=1
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0,stratify=y)
        X_train = torch.from_numpy(X_train.todense()).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test.todense()).float()
        y_test = torch.from_numpy(y_test).long()

        torch.save([X_train,y_train],f=train_file)
        torch.save([X_test,y_test],f=test_file)
    
    else: 
        X_train,y_train = torch.load(train_file)
        X_test,y_test = torch.load(test_file)
    
    if l2_norm:    
        # L2 normalize the data
        X_train /= X_train.norm(p=2,dim=1).max()
        X_test /= X_test.norm(p=2,dim=1).max()

    # X_train = torch.cat((X_train,torch.ones(X_train.size(0),1)),dim=1)
    # X_test = torch.cat((X_test,torch.ones(X_test.size(0),1)),dim=1)
    return X_train, X_test, y_train,y_test

def get_f1_score(y_true,y_pred,**kwargs):
    try:
        score = f1_score(y_true,y_pred,**kwargs)
    except Exception as e:
        # print(type(e),e)
        score = np.nan
    return score

def get_roc_score(y_true,y_pred,**kwargs):
    try:
        score = roc_auc_score(y_true,y_pred,**kwargs)
    except Exception as e:
        score = np.nan
    return score

def load_sensIT(data_dir,l2_norm=True):
    processed_dir = data_dir/"SensIT"/"processed"
    train_file = processed_dir/"training.pt"
    test_file = processed_dir/"test.pt"
    
    if not test_file.exists() or not train_file.exists():
        processed_dir.mkdir(parents=True,exist_ok=True)
        yX = pd.read_csv(data_dir/"sensIT"/"raw"/"vehicle_sensIT.csv")
        # convert sparse ARFF format
        yX = (yX.applymap(lambda x: x.split()[1].strip("{}"))).apply(pd.to_numeric)
        X = yX.values[:,1:]
        y = yX.values[:,0]
        # convert y from -1,1 to 0,1
        y = (y+1)//2
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0,stratify=y)
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).long()

        torch.save([X_train,y_train],f=train_file)
        torch.save([X_test,y_test],f=test_file)
    else:
        X_train,y_train = torch.load(train_file)
        X_test,y_test = torch.load(test_file)
    
    if l2_norm:    
        # L2 normalize the data
        X_train /= X_train.norm(p=2,dim=1).max()
        X_test /= X_test.norm(p=2,dim=1).max()
    return X_train, X_test, y_train,y_test


def load_higgs(data_dir,l2_norm=True):
    processed_dir = data_dir/"HIGGS"/"processed"
    train_file = processed_dir/"training.pt"
    test_file = processed_dir/"test.pt"
    
    if not test_file.exists() or not train_file.exists():
        print("Loading HIGGS datafile")
        processed_dir.mkdir(parents=True,exist_ok=True)
        X,y = load_svmlight_file(str((data_dir/"HIGGS"/"raw"/"HIGGS").resolve()))
        print("Preprocessing Datafile")
        # scale 
        X = MinMaxScaler().fit_transform(X.todense())
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0,stratify=y)
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).long()
        print("Saving Processed Pytorch Tensors")
        torch.save([X_train,y_train],f=train_file)
        torch.save([X_test,y_test],f=test_file)
    else:
        X_train,y_train = torch.load(train_file)
        X_test,y_test = torch.load(test_file)
    
    if l2_norm:    
        # L2 normalize the data
        X_train /= X_train.norm(p=2,dim=1).max()
        X_test /= X_test.norm(p=2,dim=1).max()
    
    # Add constant for bias
    X_train = torch.cat((X_train,torch.ones(X_train.size(0),1)),dim=1)
    X_test = torch.cat((X_test,torch.ones(X_test.size(0),1)),dim=1)
    return X_train, X_test, y_train,y_test

def load_cifar(data_dir,l2_norm=True,ovr=False,chosen_classes=[3,8]):
    processed_dir = data_dir/"CIFAR"/"processed"
    train_file = processed_dir/"training.pt"
    test_file = processed_dir/"test.pt"

    if not test_file.exists() or not train_file.exists():
        processed_dir.mkdir(parents=True,exist_ok=True)
        print("Loading CIFAR10 training and testing data")
        train_raw_file = data_dir/"CIFAR"/"raw"/"cifar10.bz2"
        test_raw_file = data_dir/"CIFAR"/"raw"/"cifar10.t.bz2"
        X_train,y_train = load_svmlight_file(str(train_raw_file))
        X_test,y_test = load_svmlight_file(str(test_raw_file))
        # scale from 0-255 to 0-1
        X_train /= 255
        X_test /= 255
        # convert to tensors
        X_train = torch.from_numpy(X_train.todense()).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test.todense()).float()
        y_test = torch.from_numpy(y_test).long()
        print("Saving Processed Pytorch Tensors")
        torch.save([X_train,y_train],f=train_file)
        torch.save([X_test,y_test],f=test_file)        
    else:
        X_train,y_train = torch.load(train_file)
        X_test,y_test = torch.load(test_file)
    

    if l2_norm:    
        # Alternate L2 norm with the max norm of the dataset
        X_train /= X_train.norm(p=2,dim=1).max()
        X_test /= X_test.norm(p=2,dim=1).max()
    
    if not ovr : 
        train_mask = np.isin(y_train,chosen_classes)
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]

        test_mask = np.isin(y_test,chosen_classes)
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        
        # convert classes 3(cat) and 5(dog) to 0 and 1
        binarizer = LabelBinarizer()
        binarizer.fit(y_train)
        y_train = torch.from_numpy(binarizer.transform(y_train).flatten())
        y_test = torch.from_numpy(binarizer.transform(y_test).flatten())
    else:
        y_train = F.one_hot(y_train)
        y_test = F.one_hot(y_test)
    

    # Add constant for bias
    X_train = torch.cat((X_train,torch.ones(X_train.size(0),1)),dim=1)
    X_test = torch.cat((X_test,torch.ones(X_test.size(0),1)),dim=1)
    return X_train,X_test,y_train,y_test

def load_epsilon(data_dir,l2_norm=True):
    processed_dir = data_dir/"epsilon"/"processed"
    train_file = processed_dir/"training.pt"
    test_file = processed_dir/"test.pt"

    if not test_file.exists() or not train_file.exists():
        processed_dir.mkdir(parents=True,exist_ok=True)
        print("Loading Epsilon training and testing data")
        train_raw_file = data_dir/"epsilon"/"raw"/"epsilon.bz2"
        test_raw_file = data_dir/"epsilon"/"raw"/"epsilon.t.bz2"
        X_train,y_train = load_svmlight_file(str(train_raw_file))
        X_test,y_test = load_svmlight_file(str(test_raw_file))
        # convert -1,1 to 1,1
        y_train = (y_train+1)/2
        y_test = (y_test+1)/2
        # convert to tensors
        X_train = torch.from_numpy(X_train.todense()).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test.todense()).float()
        y_test = torch.from_numpy(y_test).long()
        print("Saving Processed Pytorch Tensors")
        torch.save([X_train,y_train],f=train_file)
        torch.save([X_test,y_test],f=test_file)        
    else:
        X_train,y_train = torch.load(train_file)
        X_test,y_test = torch.load(test_file)
    

    if l2_norm:    
        # Alternate L2 norm with the max norm of the dataset
        X_train /= X_train.norm(p=2,dim=1).max()
        X_test /= X_test.norm(p=2,dim=1).max()
    
    # Add constant for bias
    X_train = torch.cat((X_train,torch.ones(X_train.size(0),1)),dim=1)
    X_test = torch.cat((X_test,torch.ones(X_test.size(0),1)),dim=1)
    return X_train,X_test,y_train,y_test
