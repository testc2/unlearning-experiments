'''
Created on Jan 3, 2020


'''
from torchvision.datasets.mnist import *
import torchvision.transforms as transforms
import os, sys
from pathlib import Path
from torch import nn, optim
import torch

import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader



sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/data_IO')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+'/Interpolation')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.datasets import load_svmlight_file
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import *
try:
    from data_IO.Load_data import *
    from Models.DNN_transfer import *

except ImportError:
    from Load_data import *
    from Models.DNN_transfer import *

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
    def __init__(self, samples):
        
        self.data = samples
        
    def __getitem__(self, index):
        data, target = self.data[index]
        
#         data = data.view(-1)
        
        # Your transformations here (or set it in CIFAR10)
        
        return data.type(torch.DoubleTensor), target, index

    def __len__(self):
        return len(self.data)

class Data_preparer:
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        return
    
    
    
#     def prepare_MNIST(self):
#         
#         
#         configs = load_config_data(config_file)
#     
# #     print(configs)
#         git_ignore_folder = configs['git_ignore_folder']
#         
#         train_data = MNIST(git_ignore_folder + '/mnist',
#                    download=True,
#                    transform=transforms.Compose([
# #                        transforms.Resize((32, 32)),
#                        transforms.ToTensor()]))
#         
#         
# #         for i in range(len(self.data.transforms.transform.transforms)):
#         train_X = train_data.transforms.transform.transforms[0](train_data.data.data.numpy())
#         
#         train_X = train_X.transpose(0,1).transpose(1,2)
#         
#         train_Y = train_data.targets
#         
#         
#         test_data = MNIST(git_ignore_folder + '/mnist',
#                       train=False,
#                       download=True,
#                       transform=transforms.Compose([
# #                           transforms.Resize((32, 32)),
#                           transforms.ToTensor()]))
#         
#         
#         test_X = test_data.transforms.transform.transforms[0](test_data.data.data.numpy())
#         
#         test_X = test_X.transpose(0,1).transpose(1,2)
#         
#         test_Y = test_data.targets
#         
# #         test_data.data.data = test_data.data.transforms.transform.transforms[0](test_data.data.data)
#         
#         return train_X.type(torch.DoubleTensor).view(train_X.shape[0], -1), train_Y, test_X.type(torch.DoubleTensor).view(test_X.shape[0], -1), test_Y
#         
#     
#     
#     def get_hyperparameters_MNIST(self, parameters, init_lr, regularization_rate):
#     
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
#         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
#         
# #         regularization_rate = 0.1
#         
#         return criterion, optimizer, lr_scheduler
#     
    
    def prepare_MNIST2(self):
        
        
        resnet = models.resnet50(pretrained=True)
        # freeze all model parameters
        for param in resnet.parameters():
            param.requires_grad = False
            
            
#         print(resnet.fc.in_features)
        
        
        mnist = MNIST(git_ignore_folder + '/mnist', download=True, train=False).train_data.float()
        
#         data_transform = Compose([ Resize((224, 224)),ToTensor(), Normalize((mnist.float().mean()/255,), (mnist.float().std()/255,))])
        
        data_transform = Compose([ToPILImage(), Resize((224, 224)),ToTensor()])

        train_data = MNIST(git_ignore_folder + '/mnist',
                   download=True,
                   transform=data_transform)
        
        test_data = MNIST(git_ignore_folder + '/mnist',
                      train=False,
                      download=True,
                      transform=data_transform)
        
        
#         train_X = train_data.transforms.transform.transforms[0](train_data.data.data.numpy())
        
#         train_X = data_transform(train_data)

        train_X = self.compose_train_test_data(train_data, resnet)

#         train_X = train_data.transforms.transform(transforms.ToPILImage()(train_data.data.float()))
        
        train_X = train_X.transpose(0,1).transpose(1,2)
        
        train_Y = train_data.targets
        
        
        test_data = MNIST(git_ignore_folder + '/mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
#                           transforms.Resize((32, 32)),
                          transforms.ToTensor()]))
        
        
#         test_X = test_data.transforms.transform.transforms[0](test_data.data.data.numpy())
        test_X = data_transform(test_data.data.data.numpy())
        
        test_X = test_X.transpose(0,1).transpose(1,2)
        
        test_Y = test_data.targets
        
        return train_X.type(torch.DoubleTensor).view(train_X.shape[0], 1, train_X.shape[1], train_X.shape[2]), train_Y.view(train_X.shape[0],-1), test_X.type(torch.DoubleTensor).view(test_X.shape[0], 1, test_X.shape[1], test_X.shape[2]), test_Y.view(test_X.shape[0], -1)
        
    
    
    def get_hyperparameters_MNIST2(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.MSELoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer, lr_scheduler
    
    def prepare_MNIST3(self):
        train_data = MNIST(git_ignore_folder + '/mnist',
                   download=True,
                   transform=transforms.Compose([
#                        transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
        
        test_data = MNIST(git_ignore_folder + '/mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
#                           transforms.Resize((32, 32)),
                          transforms.ToTensor()]))
        
        return train_data, test_data
        
    
    def get_hyperparameters_MNIST3(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer, lr_scheduler    


    def prepare_MNIST4(self):
        train_data = MNIST(git_ignore_folder + '/mnist',
                   download=True,
                   transform=transforms.Compose([
#                        transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
        
        test_data = MNIST(git_ignore_folder + '/mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
#                           transforms.Resize((32, 32)),
                          transforms.ToTensor()]))
        
        return train_data, test_data
        
    
    
    def get_hyperparameters_MNIST4(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.MSELoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer, lr_scheduler

    
    def prepare_MNIST(self, git_ignore_folder):
        
#         configs = load_config_data(config_file)
#     
# #     print(configs)
#         git_ignore_folder = configs['git_ignore_folder']
        
        
        train_data = MNIST(git_ignore_folder + '/mnist',
                   download=True,
                   transform=transforms.Compose([
#                         transforms.Resize((32, 32)),
                       transforms.ToTensor()]))

#         test_data = MNIST(git_ignore_folder + '/mnist',
#                       train=False,
#                       download=True,
#                       transform=transforms.Compose([
# #                         transforms.Resize((32, 32)),
#                           transforms.ToTensor()]))
        
        train_X = train_data.transforms.transform.transforms[0](train_data.data.data.numpy())
        
        train_X = train_X.transpose(0,1).transpose(1,2)
        
        train_Y = train_data.targets
        
        
        test_data = MNIST(git_ignore_folder + '/mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
#                           transforms.Resize((32, 32)),
                          transforms.ToTensor()]))
        
        
        test_X = test_data.transforms.transform.transforms[0](test_data.data.data.numpy())
        
        test_X = test_X.transpose(0,1).transpose(1,2)
        
        test_Y = test_data.targets
        
        
        train_X = train_X.reshape([train_X.shape[0], -1])
        
        
        train_X = train_X.type(torch.DoubleTensor)
        
        test_X = test_X.type(torch.DoubleTensor)
        
        test_X = test_X.reshape([test_X.shape[0], -1])
        
        mean = train_X.mean()
        std = train_X.std()
        train_X = (train_X-mean)/std
        test_X = (test_X-mean)/std
        # train_X /= train_X.norm(p=2,dim=1).unsqueeze(1)
        # test_X /= test_X.norm(p=2,dim=1).unsqueeze(1)
        train_X /= train_X.norm(p=2,dim=1).max()
        test_X /= test_X.norm(p=2,dim=1).max()
#         print(train_X.shape)

        
#         train_X = extended_by_constant_terms(train_X, False)
#         
#         test_X = extended_by_constant_terms(test_X, False)
        
        torch.save(train_X, git_ignore_folder + 'noise_X')
        
        torch.save(train_Y, git_ignore_folder + 'noise_Y')
        
        
        return train_X, train_Y, test_X, test_Y
        
    
    
    def get_hyperparameters_MNIST(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.NLLLoss(reduction="mean")
        optimizer = optim.SGD(parameters, lr = init_lr, weight_decay = regularization_rate)
#         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer
    
    def get_num_class_MNIST(self):
        return 10
    
    def prepare_MNISTOVR(self, git_ignore_folder):
        
        data_dir = git_ignore_folder
        X_train,y_train = torch.load(data_dir + "/processed/training.pt")
        X_test,y_test = torch.load(data_dir+ "/processed/test.pt")
        X_train = X_train.double().reshape(X_train.size(0),-1)
        X_test = X_test.double().reshape(X_test.size(0),-1)
        mean = X_train.mean()
        std = X_train.std()
        X_train = (X_train-mean)/std
        X_test = (X_test-mean)/std
        # L2 normalize the data
        # X_train /= X_train.norm(p=2,dim=1).unsqueeze(1)
        # X_test /= X_test.norm(p=2,dim=1).unsqueeze(1)
        # Alternate L2 norm with the max norm of the dataset
        X_train /= X_train.norm(p=2,dim=1).max()
        X_test /= X_test.norm(p=2,dim=1).max()
        train_Y = torch.nn.functional.one_hot(y_train)
        test_Y = torch.nn.functional.one_hot(y_test)
        
        return X_train, train_Y, X_test, test_Y
        

    def get_hyperparameters_MNISTOVR(self, parameters, init_lr, regularization_rate):
        # Binary loss as it is OVR
        criterion = nn.BCEWithLogitsLoss(reduction="mean") 
        optimizer = optim.SGD(parameters, lr = init_lr, weight_decay = regularization_rate)
#         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer
    
    def get_num_class_MNISTOVR(self):
        return 10
    
    
    def prepare_MNISTBinary(self, git_ignore_folder):
        data_dir = git_ignore_folder
        X_train,y_train = torch.load(data_dir + "/processed/training.pt")
        X_test,y_test = torch.load(data_dir+ "/processed/test.pt")
        X_train = X_train.double().reshape(X_train.size(0),-1)
        X_test = X_test.double().reshape(X_test.size(0),-1)
        mean = X_train.mean()
        std = X_train.std()
        X_train = (X_train-mean)/std
        X_test = (X_test-mean)/std
        # L2 normalize the data
        # X_train /= X_train.norm(p=2,dim=1).unsqueeze(1)
        # X_test /= X_test.norm(p=2,dim=1).unsqueeze(1)
        # Alternate L2 norm with the max norm of the dataset
        X_train /= X_train.norm(p=2,dim=1).max()
        X_test /= X_test.norm(p=2,dim=1).max()
        
        chosen_classes = [3,8]

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
        

        return X_train,y_train,X_test,y_test

    def get_hyperparameters_MNISTBinary(self, parameters, init_lr, regularization_rate):
        criterion = nn.BCEWithLogitsLoss(reduction="mean") 
        optimizer = optim.SGD(parameters, lr = init_lr, weight_decay = regularization_rate)
#         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer

    def get_num_class_MNISTBinary(self):
        return 2
    
    def prepare_COVTYPEBinary(self,git_ignore_folder,l2_norm=True):
        git_ignore_folder = Path(git_ignore_folder).resolve()
        processed_dir = git_ignore_folder/"processed"
        train_file = processed_dir/"binary_training.pt"
        test_file = processed_dir/"binary_test.pt"

        if not test_file.exists() or not train_file.exists():
            processed_dir.mkdir(parents=True,exist_ok=True)
            X,y = load_svmlight_file(str((git_ignore_folder/"raw"/"covtype.libsvm.binary.scale").resolve()))
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
        return X_train.double(),y_train,X_test.double(),y_test

    def get_hyperparameters_COVTYPEBinary(self, parameters, init_lr, regularization_rate):
        criterion = nn.BCEWithLogitsLoss(reduction="mean") 
        optimizer = optim.SGD(parameters, lr = init_lr, weight_decay = regularization_rate)
        
        return criterion, optimizer
    
    def get_num_class_COVTYPEBinary(self):
        return 2    

    def prepare_covtype(self,git_ignore_folder):
        
        configs = load_config_data(config_file)
    
#     print(configs)
        # git_ignore_folder = configs['git_ignore_folder']
        
        directory_name = configs['directory']
        
        train_X, train_Y, test_X, test_Y = load_data_multi_classes(True, directory_name + "covtype")
        
        train_Y = train_Y.view(-1)
        
        test_Y = test_Y.view(-1)
        
        train_X = extended_by_constant_terms(train_X, False)
        
        test_X = extended_by_constant_terms(test_X, False)
        
        torch.save(train_X, git_ignore_folder + 'noise_X')
        
        torch.save(train_Y, git_ignore_folder + 'noise_Y')
        
#         train_data = MNIST(git_ignore_folder + '/mnist',
#                    download=True,
#                    transform=transforms.Compose([
# #                         transforms.Resize((32, 32)),
#                        transforms.ToTensor()]))
#         
#         test_data = MNIST(git_ignore_folder + '/mnist',
#                       train=False,
#                       download=True,
#                       transform=transforms.Compose([
# #                         transforms.Resize((32, 32)),
#                           transforms.ToTensor()]))
        
        return train_X, train_Y.type(torch.LongTensor), test_X, test_Y.type(torch.LongTensor)
        
    
    
    def get_hyperparameters_covtype(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer
    
    def prepare_HIGGS(self,git_ignore_folder,l2_norm=True):
        data_dir = Path(git_ignore_folder).resolve()
        processed_dir = data_dir/"processed"
        train_file = processed_dir/"training.pt"
        test_file = processed_dir/"test.pt"
        X_train,y_train = torch.load(train_file)
        X_test,y_test = torch.load(test_file)

        if l2_norm:    
            # L2 normalize the data
            X_train /= X_train.norm(p=2,dim=1).max()
            X_test /= X_test.norm(p=2,dim=1).max()
        # Add constant for bias
        X_train = torch.cat((X_train,torch.ones(X_train.size(0),1)),dim=1)
        X_test = torch.cat((X_test,torch.ones(X_test.size(0),1)),dim=1)  

        return  X_train.double(),y_train,X_test.double(),y_test
    
    def get_hyperparameters_HIGGS(self, parameters, init_lr, regularization_rate):
        criterion = nn.BCEWithLogitsLoss(reduction="mean") 
        optimizer = optim.SGD(parameters, lr = init_lr, weight_decay = regularization_rate)
        
        return criterion, optimizer
    
    def get_num_class_HIGGS(self):
        return 2
    
    def prepare_CIFARBinary(self,git_ignore_folder,l2_norm=True):
        data_dir = Path(git_ignore_folder).resolve()
        processed_dir = data_dir/"processed"
        train_file = processed_dir/"training.pt"
        test_file = processed_dir/"test.pt"
        X_train,y_train = torch.load(train_file)
        X_test,y_test = torch.load(test_file)
        if l2_norm:    
            # Alternate L2 norm with the max norm of the dataset
            X_train /= X_train.norm(p=2,dim=1).max()
            X_test /= X_test.norm(p=2,dim=1).max()
        
        chosen_classes=[3,8]
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
        # Add constant for bias
        X_train = torch.cat((X_train,torch.ones(X_train.size(0),1)),dim=1)
        X_test = torch.cat((X_test,torch.ones(X_test.size(0),1)),dim=1)

        return  X_train.double(),y_train,X_test.double(),y_test
    
    def get_hyperparameters_CIFARBinary(self, parameters, init_lr, regularization_rate):
        criterion = nn.BCEWithLogitsLoss(reduction="mean") 
        optimizer = optim.SGD(parameters, lr = init_lr, weight_decay = regularization_rate)
        
        return criterion, optimizer
    
    def get_num_class_CIFARBinary(self):
        return 2    

    def prepare_EPSILON(self,git_ignore_folder,l2_norm=True):
        data_dir = Path(git_ignore_folder).resolve()
        processed_dir = data_dir/"processed"
        train_file = processed_dir/"training.pt"
        test_file = processed_dir/"test.pt" 
        X_train,y_train = torch.load(train_file)
        X_test,y_test = torch.load(test_file)
        if l2_norm:    
            # Alternate L2 norm with the max norm of the dataset
            X_train /= X_train.norm(p=2,dim=1).max()
            X_test /= X_test.norm(p=2,dim=1).max()
        # Add constant for bias
        X_train = torch.cat((X_train,torch.ones(X_train.size(0),1)),dim=1)
        X_test = torch.cat((X_test,torch.ones(X_test.size(0),1)),dim=1)  

        return  X_train.double(),y_train,X_test.double(),y_test
    
    def get_hyperparameters_EPSILON(self, parameters, init_lr, regularization_rate):
        criterion = nn.BCEWithLogitsLoss(reduction="mean") 
        optimizer = optim.SGD(parameters, lr = init_lr, weight_decay = regularization_rate)
        
        return criterion, optimizer
    
    def get_num_class_EPSILON(self):
        return 2    
    
    def prepare_higgs(self,git_ignore_folder):
        
        configs = load_config_data(config_file)
    
#     print(configs)
        # git_ignore_folder = configs['git_ignore_folder']
        
        directory_name = configs['directory']
        
        train_X, train_Y, test_X, test_Y = load_data_multi_classes(True, directory_name + "HIGGS", -500000)
        
        train_Y = train_Y.view(-1)
        
        test_Y = test_Y.view(-1)
        
        train_X = extended_by_constant_terms(train_X, False)
        
        test_X = extended_by_constant_terms(test_X, False)
        
        torch.save(train_X, git_ignore_folder + 'noise_X')
        
        torch.save(train_Y, git_ignore_folder + 'noise_Y')
        
        
        print(train_X.shape)
        
        print(test_X.shape)
#         train_data = MNIST(git_ignore_folder + '/mnist',
#                    download=True,
#                    transform=transforms.Compose([
# #                         transforms.Resize((32, 32)),
#                        transforms.ToTensor()]))
#         
#         test_data = MNIST(git_ignore_folder + '/mnist',
#                       train=False,
#                       download=True,
#                       transform=transforms.Compose([
# #                         transforms.Resize((32, 32)),
#                           transforms.ToTensor()]))
        
        return train_X, train_Y.type(torch.LongTensor), test_X, test_Y.type(torch.LongTensor)
        
    
    
    def get_hyperparameters_higgs(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer
    
    def get_num_class_higgs(self):
        return 2
    
    def get_num_class_covtype(self):
        return 7
    
    
    
    def prepare_rcv1(self,git_ignore_folder):
        
        configs = load_config_data(config_file)
    
#     print(configs)
        
        directory_name = configs['directory']
        
        X_train, y_train = load_svmlight_file(directory_name + "rcv1_train.binary")
        
        
        train_X = torch.from_numpy(X_train.todense()).type(torch.DoubleTensor)
        
        train_Y = torch.from_numpy(y_train).type(torch.DoubleTensor).view(y_train.shape[0], -1)
        
        train_Y = (train_Y + 1)/2
        
#         train_X, train_Y, test_X, test_Y = load_data_multi_classes_rcv1()
        
#         train_X, train_Y = load_data_multi_classes_single(True, directory_name + "rcv1_test.multiclass")
#         
#         test_X, test_Y = load_data_multi_classes_single(True, directory_name + "rcv1_train.multiclass")
#         
# #         train_X, train_Y, test_X, test_Y = load_data_multi_classes(True, "../../../data/covtype")
#         
#         
#         train_X = extended_by_constant_terms(train_X, False)
#         
#         test_X = extended_by_constant_terms(test_X, False)
        
#         torch.save(train_X, git_ignore_folder + 'noise_X')
#         
#         torch.save(train_Y, git_ignore_folder + 'noise_Y')
#         train_data = MNIST(git_ignore_folder + '/mnist',
#                    download=True,
#                    transform=transforms.Compose([
# #                         transforms.Resize((32, 32)),
#                        transforms.ToTensor()]))
#         
#         test_data = MNIST(git_ignore_folder + '/mnist',
#                       train=False,
#                       download=True,
#                       transform=transforms.Compose([
# #                         transforms.Resize((32, 32)),
#                           transforms.ToTensor()]))
        
        return train_X, train_Y.type(torch.LongTensor), train_X, train_Y.type(torch.LongTensor)
        
    
    
    def get_hyperparameters_rcv1(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer
    
    def get_num_class_rcv1(self):
        return 2
    
    
    
    def prepare_FashionMNIST(self):
        train_data = FashionMNIST(git_ignore_folder + '/fashion_mnist',
                   download=True,
                   transform=transforms.Compose([
#                         transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
        
        test_data = FashionMNIST(git_ignore_folder + '/fashion_mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
#                         transforms.Resize((32, 32)),
                          transforms.ToTensor()]))
        
        return train_data, test_data
    
    
    
    def get_hyperparameters_FashionMNIST(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer, lr_scheduler
    
    
    def prepare_FashionMNIST2(self):
        train_data = FashionMNIST(git_ignore_folder + '/fashion_mnist',
                   download=True,
                   transform=transforms.Compose([
#                         transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
        
        test_data = FashionMNIST(git_ignore_folder + '/fashion_mnist',
                      train=False,
                      download=True,
                      transform=transforms.Compose([
#                         transforms.Resize((32, 32)),
                          transforms.ToTensor()]))
        
        return train_data, test_data
    
    
    
    def get_hyperparameters_FashionMNIST2(self, parameters, init_lr, regularization_rate):
    
        criterion = nn.MSELoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer, lr_scheduler
    
    
    def prepare_cifar10(self):
        
        
        configs = load_config_data(config_file)
    
#     print(configs)
        git_ignore_folder = configs['git_ignore_folder']
        
        directory_name = configs['directory']
        
        X_train, y_train = load_svmlight_file(directory_name + "cifar10")
        
        X_test, y_test = load_svmlight_file(directory_name + "cifar10.t")
        
        
        train_X = torch.from_numpy(X_train.todense()).type(torch.DoubleTensor)/255
        
        train_Y = torch.from_numpy(y_train).type(torch.LongTensor).view(y_train.shape[0], -1)
        
        test_X = torch.from_numpy(X_test.todense()).type(torch.DoubleTensor)/255
        
        test_Y = torch.from_numpy(y_test).type(torch.LongTensor).view(y_test.shape[0], -1)
        
        
        
#         transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 
#     
#         data_train = torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
#                        download=True,
#                        transform=transform)
#         data_test = torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
#                           train=False,
#                           download=True,
#                           transform=transform)

#         data_train = torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
#                    download=True,
#                    transform=transforms.Compose([
#                        transforms.Resize((32, 32)),
#                        transforms.RandomHorizontalFlip(),
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                        ]))
#         data_test = torchvision.datasets.CIFAR10(git_ignore_folder+ '/cifar10',
#                       train=False,
#                       download=True,
#                       transform=transforms.Compose([
#                           transforms.ToTensor(),
#                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                           ]))

        
        return train_X, train_Y, test_X, test_Y
        
    
    
    def get_hyperparameters_cifar10(self, parameters, init_lr, regularization_rate):
    
    
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer, lr_scheduler
    
    
    def get_num_class_cifar10(self):
        return 10
    
    def compute_output_before_last_layer(self, input, tl, transfer_model, transfer_model_modules, get_transfer_model_func, last_layer, is_GPU, device):
        
        this_input = input.clone()
        
        if is_GPU:
            this_input = this_input.to(device)
#         expect_output = model.forward(input)
#         for i in range(len(list(model.children()))-1):
#         for i in range(len(transfer_model_modules) - 1):
#             output = transfer_model_modules[i].double()(this_input)
#             
#             del this_input
#             
#             this_input = output
        
        
        output = get_transfer_model_func(tl, transfer_model, this_input)
        
        del this_input
        
        
        
#         if len(output.shape) > 2 or (not output.shape[1] == in_feature_num):
#             output = torch.flatten(this_input, 1)
#             
#             del this_input
        
        expect_output = last_layer.double()(output)
        
        
        output_cpu = output.to('cpu') 
        
        expect_output_cpu =expect_output.to('cpu')
        
        del output, expect_output
        
        
        return output_cpu,expect_output_cpu 
            
            
        
    
    def compose_train_test_data(self, data_train, resnet):
        
        train_X = []
        
        for i in range(data_train.data.shape[0]):
            curr_train_X = data_train.transforms.transform(data_train.data[i])
            
            curr_transformed_X, _ = self.compute_output_before_last_layer(curr_train_X.view(1, curr_train_X.shape[0], curr_train_X.shape[1], curr_train_X.shape[2]), resnet)
            
#             curr_transformed_X = resnet(curr_train_X.view(1, curr_train_X.shape[0], curr_train_X.shape[1], curr_train_X.shape[2]))
#             
#             print(i)
            
            train_X.append(curr_transformed_X)
            
        return torch.stack(train_X, 0)
            
    
    
    def normalize(self, data):
    
        print('normalization start!!')
        
        x_max,_ = torch.max(data, axis = 0)
        
        x_min,_ = torch.min(data, axis = 0)
        
        range = x_max - x_min
        
        update_data = data[:,range != 0] 
        
        
    #     print(average_value.shape)
    #     
    #     print(data)
    #     
    #     print(average_value)
    #     
    #     print(std_value)
        
        data = (update_data - x_min[range!=0])/range[range!=0]
        
    #     data = data /std_value
        
        return data
    
    def construct_full_X_Y(self, dataloader, transfer_model, transfer_model_modules, transfer_model_name, is_GPU, device):
        
        full_features = []
        
        full_labels = []
        
        i = 0
        
        
        
        get_transfer_model_func = getattr(Transfer_learning, "compute_before_last_layer_" + transfer_model_name)
        
        get_last_layer_func = getattr(Transfer_learning, "get_last_layer_" + transfer_model_name)
        
        
        tl = Transfer_learning()


        last_layer = get_last_layer_func(tl, transfer_model)
        
        
        for features, labels, ids in dataloader:
            
            print(i, ids.shape[0])
            
            
            transfered_features,_ = self.compute_output_before_last_layer(features, tl, transfer_model, transfer_model_modules, get_transfer_model_func, last_layer, is_GPU, device)
            
            full_features.append(transfered_features)
            
            print(transfered_features.shape)
            
            full_labels.append(labels)
            
            i+=1
            
            
        full_X = torch.cat(full_features, 0)
        
        full_Y = torch.cat(full_labels, 0)
        
        print(full_X.shape)
        
        full_X = self.normalize(full_X)
        
        return full_X, full_Y
            
        
        
        
        
    
        
    
    def prepare_cifar10_2(self, transfer_model, transfer_model_name, is_GPU, device):
        
        
#         configs = load_config_data(config_file)
#     
# #     print(configs)
#         git_ignore_folder = configs['git_ignore_folder']
#         
#         directory_name = configs['directory']
#         
#         X_train, y_train = load_svmlight_file(directory_name + "cifar10")
#         
#         X_test, y_test = load_svmlight_file(directory_name + "cifar10.t")
#         
#         
#         train_X = torch.from_numpy(X_train.todense()).type(torch.DoubleTensor)/255
#         
#         train_Y = torch.from_numpy(y_train).type(torch.LongTensor).view(y_train.shape[0], -1)
#         
#         test_X = torch.from_numpy(X_test.todense()).type(torch.DoubleTensor)/255
#         
#         test_Y = torch.from_numpy(y_test).type(torch.LongTensor).view(y_test.shape[0], -1)
        
        
#         resnet = models.resnet50(pretrained=True)
#         # freeze all model parameters
#         for param in resnet.parameters():
#             param.requires_grad = False
#         transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         transform = transforms.Compose([ToPILImage(), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
     
        data_train = torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
                       download=True,
                       transform=transform)
        data_test = torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
                          train=False,
                          download=True,
                          transform=transform)
        
        
        
        if is_GPU:
            transfer_model.to(device)

        train_dataset = MyDataset(data_train)
        
        test_dataset = MyDataset(data_test)

        data_train_loader = DataLoader(train_dataset, batch_size=100, num_workers=0)
        data_test_loader = DataLoader(test_dataset, batch_size=100, num_workers=0)
        
        
        transfer_model_modules = list(transfer_model.children())
        
        train_X, train_Y = self.construct_full_X_Y(data_train_loader, transfer_model, transfer_model_modules, transfer_model_name, is_GPU, device)
        
        test_X, test_Y = self.construct_full_X_Y(data_test_loader, transfer_model, transfer_model_modules, transfer_model_name, is_GPU, device)
        


#         train_X = data_train.transforms.transform(data_train.data)
#         train_X = self.compose_train_test_data(data_train, transfer_model)
#         
#         train_Y = data_train.targets
#         
#         
#         test_X = self.compose_train_test_data(data_test, transfer_model)
#         
#         test_Y = data_test.targets

#         data_train = torchvision.datasets.CIFAR10(git_ignore_folder + '/cifar10',
#                    download=True,
#                    transform=transforms.Compose([
#                        transforms.Resize((32, 32)),
#                        transforms.RandomHorizontalFlip(),
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                        ]))
#         data_test = torchvision.datasets.CIFAR10(git_ignore_folder+ '/cifar10',
#                       train=False,
#                       download=True,
#                       transform=transforms.Compose([
#                           transforms.ToTensor(),
#                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                           ]))

        
        return train_X, train_Y, test_X, test_Y
        
    
    
    def get_hyperparameters_cifar10_2(self, parameters, init_lr, regularization_rate):
    
    
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
#         regularization_rate = 0.1
        
        return criterion, optimizer, lr_scheduler
    
    
    def get_num_class_cifar10_2(self):
        return 10
    
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.SGD(parameters, lr=init_lr, weight_decay = regularization_rate)
#         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
#         
# #         regularization_rate = 0.1
#         
#         return criterion, optimizer, lr_scheduler
    