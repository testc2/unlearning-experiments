'''
Created on Jun 24, 2020

'''
import sys, os
import torch
import time
import copy
import psutil


sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/data_IO')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Interpolation')
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/Models')

sys.path.append(os.path.abspath(__file__))

from Models.Data_preparer import *
from utils import *
from model_train import *

try:
    from data_IO.Load_data import *
    from utils import *
    from Models.DNN import DNNModel
    from Models.Data_preparer import *
    from Models.DNN_single import *
    from Models.DNN2 import DNNModel2
    from Models.DNN3 import DNNModel3
    from Models.ResNet import *
    from Models.Pretrained_models import *

except ImportError:
    from Load_data import *
    from utils import *
    from Models.DNN import DNNModel
    from Models.DNN2 import DNNModel2
    from Models.DNN3 import DNNModel3
    from Models.Data_preparer import *
    from Models.DNN_single import *
    from Models.ResNet import *
    from Models.Pretrained_models import *

from main_delete import model_update_deltagrad,model_update_standard_lib
from main_delete_ovr import model_update_deltagrad_ovr,model_update_standard_lib_ovr
def model_privacy_del(args, method, lr_lists):
    model_name = args.model
    git_ignore_folder = args.repo
    dataset_name = args.dataset
    num_epochs = args.epochs
    batch_size = args.bz
    is_GPU = args.GPU
    regularization_coeff = args.wd
    
    if not is_GPU:
        device = torch.device("cpu")
    else:    
        GPU_ID = int(args.GID)
        device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    
    model_class = getattr(sys.modules[__name__], model_name)
    
    data_preparer = Data_preparer()
    
    dataset_train = torch.load(git_ignore_folder + "dataset_train")
    
    dataset_test = torch.load(git_ignore_folder + "dataset_test")
    
    
    delta_data_ids = torch.load(git_ignore_folder + "delta_data_ids")
    
    if len(delta_data_ids):
        X_remove = dataset_train.data[delta_data_ids]
        y_remove = dataset_train.labels[delta_data_ids]
        dataset_remove = dataset_train.__class__(X_remove,y_remove)
        remain_ids = torch.LongTensor(list(set(range(len(dataset_train)))-set(delta_data_ids)))
        X_prime = dataset_train.data[remain_ids]
        y_prime = dataset_train.labels[remain_ids]
        dataset_prime = dataset_train.__class__(X_prime,y_prime)
    else:
        X_remove = torch.tensor([])
        y_remove = torch.tensor([])
        dataset_remove = dataset_train.__class__(X_remove,y_remove)
        dataset_prime = dataset_train

    learning_rate_all_epochs = torch.load(git_ignore_folder + 'learning_rate_all_epochs')
    
    random_ids_all_epochs = torch.load(git_ignore_folder + 'random_ids_multi_epochs')
    
    
    sorted_ids_multi_epochs = torch.load(git_ignore_folder + 'sorted_ids_multi_epochs')
    
    mini_batch_num = int((len(dataset_train) - 1)/batch_size) + 1
    
    
    para_list_all_epochs = torch.load(git_ignore_folder + 'para_list_all_epochs')
    
    gradient_list_all_epochs = torch.load(git_ignore_folder + 'gradient_list_all_epochs')
    
    dim = [len(dataset_train), len(dataset_train[0][0])]
    
    origin_train_data_size = len(dataset_train)
    
    num_class = get_data_class_num_by_name(data_preparer, dataset_name)
    
    hyper_para_function=getattr(Data_preparer, "get_hyperparameters_" + dataset_name)
    
    if args.model == "LR":
        model = model_class(dim[1])
    else:
        model = model_class(dim[1], num_class)    
    
    init_para_list = list(torch.load(git_ignore_folder + 'init_para'))
    
    init_model(model,init_para_list)
    
    print('data dimension::',dim)
    
    if is_GPU:
        model.to(device)
    
    criterion, optimizer = hyper_para_function(data_preparer, model.parameters(), lr_lists[0], regularization_coeff)
    
    # noise_levels = [0,0.01,0.1,1,10,100]
    noise_levels = [0.01,0.1,1,10,100]

    if method == baseline_method:
        
        
        t1 = time.time()
                
        updated_model = model_update_standard_lib(num_epochs, dataset_train, model, random_ids_all_epochs, sorted_ids_multi_epochs, delta_data_ids, batch_size, learning_rate_all_epochs, criterion, optimizer, is_GPU, device)
    
        t2 = time.time()
            
        process = psutil.Process(os.getpid())

        print('memory usage::', process.memory_info().rss)
        
        
        print('time_baseline::', t2 - t1)
    
        origin_model = torch.load(git_ignore_folder + 'origin_model')    
    
        torch.save(updated_model, git_ignore_folder + 'model_base_line')
        
#         torch.save(exp_para_list, git_ignore_folder + 'exp_para_list')
#         
#         torch.save(exp_grad_list, git_ignore_folder + 'exp_grad_list')    
        with torch.no_grad():
            for noise_seed in range(args.num_seeds):
                for noise in noise_levels:
                    print(f'<noise sigma="{noise}" seed="{noise_seed}">')
                    model_copy = copy.deepcopy(updated_model)
                    torch.manual_seed(noise_seed)
                    for param in model_copy.parameters():
                        param.add_(torch.randn(param.size())*noise)
                    compute_model_para_diff(list(origin_model.parameters()), list(model_copy.parameters()))
                    test(model_copy,dataset_test,batch_size,criterion,len(dataset_test),is_GPU,device)
                    if X_remove.size(0)>0:
                        print("Remove ",end="")
                        test(model_copy,dataset_remove,batch_size,criterion,len(dataset_remove),is_GPU,device)
                        print("Remain ",end="")
                        test(model_copy,dataset_prime,batch_size,criterion,len(dataset_prime),is_GPU,device)
                    print("</noise>")
            
    else:
        if method == deltagrad_method:
            
            period = args.period
            init_epochs = args.init
            m = args.m
            cached_size = args.cached_size
            grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor = cache_grad_para_history(git_ignore_folder, cached_size, is_GPU, device)
            t1 = time.time()
            updated_model = model_update_deltagrad(num_epochs, period, 1, init_epochs, dataset_train, model, grad_list_all_epochs_tensor, para_list_all_epoch_tensor, grad_list_GPU_tensor, para_list_GPU_tensor, cached_size, delta_data_ids, m, learning_rate_all_epochs, random_ids_all_epochs, sorted_ids_multi_epochs, batch_size, dim, criterion, optimizer, regularization_coeff, is_GPU, device)
            t2 = time.time()
            process = psutil.Process(os.getpid())
            print('memory usage::', process.memory_info().rss)
            print('time_deltagrad::', t2 - t1)
            model_base_line = torch.load(git_ignore_folder + 'model_base_line')
            torch.save(updated_model, git_ignore_folder + 'model_deltagrad')
            with torch.no_grad():
                for noise_seed in range(args.num_seeds):    
                    for noise in noise_levels:
                        print(f'<noise sigma="{noise}" seed="{noise_seed}">')
                        model_copy = copy.deepcopy(updated_model)
                        torch.manual_seed(noise_seed)
                        for param in model_copy.parameters():
                            param.add_(torch.randn(param.size())*noise)
                        compute_model_para_diff(list(model_base_line.parameters()), list(model_copy.parameters()))
                        test(model_copy,dataset_test,batch_size,criterion,len(dataset_test),is_GPU,device)
                        print("Remove ",end="")
                        test(model_copy,dataset_remove,batch_size,criterion,len(dataset_remove),is_GPU,device)
                        print("Remain ",end="")
                        test(model_copy,dataset_prime,batch_size,criterion,len(dataset_prime),is_GPU,device)
                        print("</noise>")


def model_privacy_del_ovr(args, method, lr_lists):

    model_name = args.model

    git_ignore_folder = args.repo

    dataset_name = args.dataset

    num_epochs = args.epochs

    batch_size = args.bz

    is_GPU = args.GPU

    regularization_coeff = args.wd

    if not is_GPU:
        device = torch.device("cpu")
    else:
        GPU_ID = int(args.GID)
        device = torch.device(
            "cuda:" + str(GPU_ID) if torch.cuda.is_available() else "cpu"
        )

    model_class = getattr(sys.modules[__name__], model_name)

    data_preparer = Data_preparer()

    dataset_train = torch.load(git_ignore_folder + "dataset_train")

    dataset_test = torch.load(git_ignore_folder + "dataset_test")

    delta_data_ids = torch.load(git_ignore_folder + "delta_data_ids")
    
    if len(delta_data_ids):
        X_remove = dataset_train.data[delta_data_ids]
        y_remove = dataset_train.labels[delta_data_ids]
        dataset_remove = dataset_train.__class__(X_remove,y_remove,ovr=True)
        remain_ids = torch.LongTensor(list(set(range(len(dataset_train)))-set(delta_data_ids)))
        X_prime = dataset_train.data[remain_ids]
        y_prime = dataset_train.labels[remain_ids]
        dataset_prime = dataset_train.__class__(X_prime,y_prime,ovr=True)
    else:
        X_remove = torch.tensor([])
        y_remove = torch.tensor([])
        dataset_remove = dataset_train.__class__(X_remove,y_remove,ovr=True)
        dataset_prime = dataset_train
    
    learning_rate_all_epochs_ovr = torch.load(
        git_ignore_folder + "learning_rate_all_epochs_ovr"
    )

    random_ids_all_epochs = torch.load(git_ignore_folder + "random_ids_multi_epochs")

    sorted_ids_multi_epochs = torch.load(git_ignore_folder + "sorted_ids_multi_epochs")

    mini_batch_num = int((len(dataset_train) - 1) / batch_size) + 1

    para_list_all_epochs_ovr = torch.load(git_ignore_folder + "para_list_all_epochs_ovr")

    gradient_list_all_epochs_ovr = torch.load(
        git_ignore_folder + "gradient_list_all_epochs_ovr"
    )

    dim = [len(dataset_train), len(dataset_train[0][0])]

    origin_train_data_size = len(dataset_train)

    num_class = get_data_class_num_by_name(data_preparer, dataset_name)

    hyper_para_function = getattr(Data_preparer, "get_hyperparameters_" + dataset_name)
    
    models = []
    criterion = None
    optimizer_ovr = []
    init_para_class_list = list(torch.load(git_ignore_folder + "init_para_class"))
    noise_vectors_ovr = None
    if args.remove_method == "Guo":
        noise_vectors_ovr = torch.load(git_ignore_folder + "noise_vectors_ovr")

    for class_id in range(num_class):

        if args.model == "LR":
            model = model_class(dim[1])
        else:
            model = model_class(dim[1], num_class)

        init_para_list = init_para_class_list[class_id]
        init_model(model, init_para_list)

        if is_GPU:
            model.to(device)
        
        criterion, optimizer = hyper_para_function(
            data_preparer, model.parameters(), lr_lists[0], regularization_coeff
        )
        models.append(model)
        optimizer_ovr.append(optimizer)
    
    noise_levels = [0,0.01,0.1,1,10,100]

    if method == baseline_method:

        t1 = time.time()

        updated_models = model_update_standard_lib_ovr(
            num_epochs,
            dataset_train,
            models,
            random_ids_all_epochs,
            sorted_ids_multi_epochs,
            delta_data_ids,
            batch_size,
            learning_rate_all_epochs_ovr,
            criterion,
            optimizer_ovr,
            is_GPU,
            device,
            num_class,
            noise_vectors_ovr,
        )

        t2 = time.time()

        process = psutil.Process(os.getpid())

        print("memory usage::", process.memory_info().rss)

        print("time_baseline::", t2 - t1)

        origin_models = torch.load(git_ignore_folder + "origin_model_ovr")

        torch.save(updated_models, git_ignore_folder + "model_base_line_ovr")
        with torch.no_grad():
            for noise_seed in range(args.num_seeds):
                for noise in noise_levels:
                    print(f'<noise sigma="{noise}" seed="{noise_seed}">')
                    torch.manual_seed(noise_seed)
                    perturbed_models = []
                    for model in updated_models:
                        model_copy = copy.deepcopy(model)
                        for param in model_copy.parameters():
                            param.add_(torch.randn(param.size())*noise)
                        perturbed_models.append(model_copy)
                    
                    compute_model_para_diff_ovr(origin_models,perturbed_models)
                    test_ovr(
                        perturbed_models,
                        dataset_test,
                        batch_size,
                        criterion,
                        len(dataset_test),
                        is_GPU,
                        device,
                        num_class
                    )
                    if X_remove.size(0)>0:
                        print("Remove ", end="")
                        test_ovr(
                            perturbed_models,
                            dataset_remove,
                            batch_size,
                            criterion,
                            len(dataset_remove),
                            is_GPU,
                            device,
                            num_class
                        )
                        print("Remain ", end="")
                        test_ovr(
                            perturbed_models,
                            dataset_prime,
                            batch_size,
                            criterion,
                            len(dataset_prime),
                            is_GPU,
                            device,
                            num_class
                        )
                    print("</noise>")

    elif method == deltagrad_method:
        period = args.period
        init_epochs = args.init
        m = args.m
        cached_size = args.cached_size
        t1 = time.time()   
        updated_models = model_update_deltagrad_ovr(
            num_epochs,
            period,
            1,
            init_epochs,
            dataset_train,
            models,
            gradient_list_all_epochs_ovr,
            para_list_all_epochs_ovr,
            cached_size,
            delta_data_ids,
            m,
            learning_rate_all_epochs_ovr,
            random_ids_all_epochs,
            sorted_ids_multi_epochs,
            batch_size,
            dim,
            criterion,
            optimizer_ovr,
            regularization_coeff,
            is_GPU,
            device,
            num_classes=num_class,
            noise_vectors_ovr=noise_vectors_ovr,
        )

        t2 = time.time()

        process = psutil.Process(os.getpid())

        print("memory usage::", process.memory_info().rss)

        print("time_deltagrad::", t2 - t1)

        model_base_line = torch.load(git_ignore_folder + "model_base_line_ovr")

        torch.save(updated_models, git_ignore_folder + "model_deltagrad_ovr")
        with torch.no_grad():
            for noise_seed in range(args.num_seeds):
                for noise in noise_levels:
                    print(f'<noise sigma="{noise}" seed="{noise_seed}">')
                    torch.manual_seed(noise_seed)
                    perturbed_models = []
                    for model in updated_models:
                        model_copy = copy.deepcopy(model)
                        for param in model_copy.parameters():
                            param.add_(torch.randn(param.size())*noise)
                        perturbed_models.append(model_copy)
                    
                    compute_model_para_diff_ovr(model_base_line, perturbed_models)
                    test_ovr(
                        perturbed_models,
                        dataset_test,
                        batch_size,
                        criterion,
                        len(dataset_test),
                        is_GPU,
                        device,
                        num_class
                    )
                    print("Remove ", end="")
                    test_ovr(
                        perturbed_models,
                        dataset_remove,
                        batch_size,
                        criterion,
                        len(dataset_remove),
                        is_GPU,
                        device,
                        num_class
                    )
                    print("Remain ", end="")
                    test_ovr(
                        perturbed_models,
                        dataset_prime,
                        batch_size,
                        criterion,
                        len(dataset_prime),
                        is_GPU,
                        device,
                        num_class
                    )

                    print("</noise>")