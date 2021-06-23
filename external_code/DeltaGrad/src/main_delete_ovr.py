"""
Created on Jun 24, 2020

"""


import sys, os
import torch
import time


import psutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/data_IO")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/Interpolation")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/Models")


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.abspath(__file__))


from Models.Data_preparer import *

from utils import *

from model_train import *
from main_delete import *
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


def get_batch_train_data_ovr(dataset_train, ids, class_id):
    
    batch_x_train_cp, batch_y_train_cp = dataset_train.data[ids],dataset_train.labels[ids,class_id] 
    
    return batch_x_train_cp, batch_y_train_cp, ids

def model_update_standard_lib_ovr(
    num_epochs,
    dataset_train,
    models,
    random_ids_multi_epochs,
    sorted_ids_multi_epochs,
    delta_data_ids,
    batch_size,
    learning_rate_all_epochs_ovr,
    criterion,
    optimizer_ovr,
    is_GPU,
    device,
    num_classes,
    noise_vectors_ovr=None,
):

    elapse_time = 0

    overhead = 0
    #
    overhead3 = 0

    t1 = time.time()

    old_lr = -1

    random_ids_list_all_epochs = []

    t5 = time.time()

    # compute which samples are remaining in each batch of each epoch
    for k in range(num_epochs):

        random_ids = random_ids_multi_epochs[k]

        sort_idx = sorted_ids_multi_epochs[k]  # random_ids.numpy().argsort()

        if delta_data_ids.shape[0] > 1:
            all_indexes = np.sort(sort_idx[delta_data_ids])
        else:
            all_indexes = torch.tensor([sort_idx[delta_data_ids]])

        id_start = 0

        id_end = 0

        random_ids_list = []

        for j in range(0, len(dataset_train), batch_size):

            end_id = j + batch_size

            if end_id > len(dataset_train):
                end_id = len(dataset_train)

            # if deletes are all 0
            if delta_data_ids.shape[0] < 1:
                if all_indexes.size(1) ==0 :
                    id_end = 0
            elif all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)

            removed_ids = random_ids[all_indexes[id_start:id_end]]

            if removed_ids.shape[0] > 0:
                curr_matched_ids = get_remaining_subset_data_per_epoch(
                    random_ids[j:end_id], removed_ids
                )
            else:
                curr_matched_ids = random_ids[j:end_id]
            random_ids_list.append(curr_matched_ids)
            id_start = id_end

        random_ids_list_all_epochs.append(random_ids_list)

        t6 = time.time()

        overhead2 = t6 - t5
    for class_id in range(num_classes):
        count = 0
        optimizer = optimizer_ovr[class_id]
        model = models[class_id]

        for k in range(num_epochs):

            random_ids_list = random_ids_list_all_epochs[k]
            epoch_loss = []
            for j in range(len(random_ids_list)):

                curr_random_ids = random_ids_list[j]
                curr_matched_ids_size = len(curr_random_ids)

                # if all elements in batch are removed, skip
                if curr_matched_ids_size <= 0:

                    count += 1

                    continue

                if not is_GPU:

                    batch_X = dataset_train.data[curr_random_ids]

                    batch_Y = dataset_train.labels[curr_random_ids,class_id]

                else:
                    batch_X = dataset_train.data[curr_random_ids].to(device)

                    batch_Y = dataset_train.labels[curr_random_ids,class_id].to(device)
                
                learning_rate = learning_rate_all_epochs_ovr[class_id][count]

                if not learning_rate == old_lr:
                    update_learning_rate(optimizer, learning_rate)

                old_lr = learning_rate
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_Y)

                if noise_vectors_ovr is not None:
                    noise_loss = torch.tensor(0.0)
                    for i, param in enumerate(model.parameters()):
                        noise_loss += (param * noise_vectors_ovr[class_id][i]).sum()
                    if k % 100 == 0:
                        print(f"noise_loss : {noise_loss}", end="")
                    loss += noise_loss / batch_X.shape[0]

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.detach().cpu().item())
                count += 1

            if k == num_epochs-1:
                print(f"Class: {class_id} epoch: {k} Baseline Loss {torch.tensor(epoch_loss).mean()}")

    t2 = time.time()

    elapse_time += t2 - t1

    print("training time is", elapse_time)

    print("overhead::", overhead)

    print("overhead2::", overhead2)
    #
    print("overhead3::", overhead3)

    return models

def model_update_deltagrad_ovr(
    max_epoch,
    period,
    length,
    init_epochs,
    dataset_train,
    models,
    gradient_list_all_epochs_ovr,
    para_list_all_epochs_ovr,
    cached_size,
    delta_ids,
    m,
    learning_rate_all_epochs_ovr,
    random_ids_multi_super_iterations,
    sorted_ids_multi_super_iterations,
    batch_size,
    dim,
    criterion,
    optimizer_ovr,
    regularization_coeff,
    is_GPU,
    device,
    num_classes,
    noise_vectors_ovr=None
):
    '''function to use deltagrad for incremental updates'''
    
    '''detect which samples are removed from each mini-batch'''

    overhead3 = 0
    random_ids_list_all_epochs = []
    removed_batch_empty_list = []
    
    t5 = time.time()
    for k in range(max_epoch):
    
        random_ids = random_ids_multi_super_iterations[k]
        sort_idx = sorted_ids_multi_super_iterations[k]
        
        if delta_ids.shape[0] > 1:
            all_indexes = np.sort(sort_idx[delta_ids])
        else:
            all_indexes = torch.tensor([sort_idx[delta_ids]])
                
        id_start = 0
        id_end = 0
        random_ids_list = []
        
        for j in range(0, dim[0], batch_size):
        
            end_id = j + batch_size
            
            if end_id > dim[0]:
                end_id = dim[0]
            
            if all_indexes[-1] < end_id:
                id_end = all_indexes.shape[0]
            else:
                id_end = np.argmax(all_indexes >= end_id)
            
            curr_matched_ids = random_ids[all_indexes[id_start:id_end]]
            curr_matched_id_num = curr_matched_ids.shape[0]

            if curr_matched_id_num > 0:
                random_ids_list.append(curr_matched_ids)
                removed_batch_empty_list.append(False)
            else:
                random_ids_list.append(random_ids[0:1])
                removed_batch_empty_list.append(True)
            
            id_start = id_end
                
        random_ids_list_all_epochs.append(random_ids_list)        
    
    t6 = time.time()
    overhead3 += (t6  -t5)
    print('Detection overhead::', overhead3)
    ''' begin deltagrad for each target class'''
    updated_models = []

    for class_id in range(num_classes):
        model = models[class_id]
        optimizer = optimizer_ovr[class_id]
        learning_rate_all_epochs = learning_rate_all_epochs_ovr[class_id]
        if noise_vectors_ovr is not None:
            noise_vectors = noise_vectors_ovr[class_id]
        else:
            noise_vectors = None
        # pre-fetch history 
        (
            gradient_list_all_epochs_tensor,
            para_list_all_epochs_tensor,
            grad_list_GPU_tensor,
            para_list_GPU_tensor,
        ) = cache_grad_para_history_ovr(para_list_all_epochs_ovr, gradient_list_all_epochs_ovr, cached_size, is_GPU, device, class_id)        
        
        para = list(model.parameters())    
        use_standard_way = False
        recorded = 0
        full_shape_list, shape_list, total_shape_size = get_model_para_shape_list(model.parameters())
        if not is_GPU:
            vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double)
        else:
            vec_para_diff = torch.zeros([total_shape_size, 1], dtype = torch.double, device=device)

        S_k_list = deque()
        Y_k_list = deque()
        overhead2 = 0
        overhead4 = 0
        overhead5 = 0
        old_lr = 0
        cached_id = 0
        batch_id = 1

        '''main for loop of deltagrad'''
        
        i = 0
        for k in range(max_epoch):
        
            random_ids = random_ids_multi_super_iterations[k]
            random_ids_list = random_ids_list_all_epochs[k]
            id_start = 0
            id_end = 0
            j = 0
            curr_init_epochs = init_epochs
            for p in range(len(random_ids_list)):
                curr_matched_ids = random_ids_list[p]
                end_id = j + batch_size
                if end_id > dim[0]:
                    end_id = dim[0]        
                curr_matched_ids_size = 0
                if not removed_batch_empty_list[i]:
                    if not is_GPU:
                        batch_delta_X = dataset_train.data[curr_matched_ids]
                        batch_delta_Y = dataset_train.labels[curr_matched_ids,class_id]
                    else:
                        batch_delta_X = dataset_train.data[curr_matched_ids].to(device)
                        batch_delta_Y = dataset_train.labels[curr_matched_ids,class_id].to(device)
                    curr_matched_ids_size = len(curr_matched_ids)
                
                learning_rate = learning_rate_all_epochs[i]
                if end_id - j - curr_matched_ids_size <= 0:
                    i += 1
                    continue            
                
                if not learning_rate == old_lr:
                    update_learning_rate(optimizer, learning_rate)
                old_lr = learning_rate    
                
                if (i-curr_init_epochs)%period == 0:
                    recorded = 0
                    use_standard_way = True
                    
                if i< curr_init_epochs or use_standard_way == True:
                    t7 = time.time()
                    '''explicitly evaluate the gradient'''
                    curr_rand_ids = random_ids[j:end_id]
                    if not removed_batch_empty_list[i]:
                        curr_matched_ids2 = get_remaining_subset_data_per_epoch(curr_rand_ids, curr_matched_ids)
                    else:
                        curr_matched_ids2 = curr_rand_ids
                    
                    batch_remaining_X, batch_remaining_Y, batch_remaining_ids = get_batch_train_data_ovr(dataset_train, curr_matched_ids2, class_id)
                    if is_GPU:
                        batch_remaining_X = batch_remaining_X.to(device)
                        batch_remaining_Y = batch_remaining_Y.to(device)
                    
                    t8 = time.time()
                    overhead4 += (t8 - t7)
                  
                    t5 = time.time()
                    init_model(model, para)
                    compute_derivative_one_more_step(model, batch_remaining_X, batch_remaining_Y, criterion, optimizer, noise_vectors)
                    expect_gradients = get_all_vectorized_parameters1(model.get_all_gradient())
                    
                    t6 = time.time()
                    overhead3 += (t6 - t5)
                    
                    gradient_remaining = 0
                    if not removed_batch_empty_list[i]:
                        t3 = time.time()
                        clear_gradients(model.parameters())
                        compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer,noise_vectors)
                        gradient_remaining = get_all_vectorized_parameters1(model.get_all_gradient())     
                        t4 = time.time()
                        overhead2 += (t4  -t3)
                    
                    with torch.no_grad():
                        curr_para = get_all_vectorized_parameters1(para)
                        if k>0 or (p > 0 and k == 0):
                            prev_para = para_list_GPU_tensor[cached_id]
                            curr_s_list = (curr_para - prev_para) + 1e-16
                            S_k_list.append(curr_s_list)
                            if len(S_k_list) > m:
                                removed_s_k = S_k_list.popleft()
                                del removed_s_k
                            
                        gradient_full = (expect_gradients*curr_matched_ids2.shape[0] + gradient_remaining*curr_matched_ids_size)/(curr_matched_ids2.shape[0] + curr_matched_ids_size)
                        if k>0 or (p > 0 and k == 0):
                            Y_k_list.append(gradient_full - grad_list_GPU_tensor[cached_id] + regularization_coeff*curr_s_list+ 1e-16)
                            if len(Y_k_list) > m:
                                removed_y_k = Y_k_list.popleft()
                                del removed_y_k
                        
                        para = get_devectorized_parameters((1-learning_rate*regularization_coeff)*curr_para - learning_rate*expect_gradients, full_shape_list, shape_list)
                        recorded += 1
                        
                        del gradient_full
                        del gradient_remaining
                        del expect_gradients
                        del batch_remaining_X
                        del batch_remaining_Y
                        if not removed_batch_empty_list[i]:
                            del batch_delta_X
                            del batch_delta_Y
                        
                        if k>0 or (p > 0 and k == 0):
                            del prev_para
                            del curr_para
                        
                        if recorded >= length:
                            use_standard_way = False
                    
                else:
                    '''use l-bfgs algorithm to evaluate the gradients'''
                    gradient_dual = None
                    if not removed_batch_empty_list[i]:
                        init_model(model, para)
                        compute_derivative_one_more_step(model, batch_delta_X, batch_delta_Y, criterion, optimizer, noise_vectors)
                        gradient_dual = model.get_all_gradient()
                        
                    with torch.no_grad():
                        vec_para_diff = torch.t((get_all_vectorized_parameters1(para) - para_list_GPU_tensor[cached_id]))
                        if (i-curr_init_epochs)/period >= 1:
                            if (i-curr_init_epochs) % period == 1:
                                zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = prepare_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, is_GPU, device)
                                mat = np.linalg.inv(mat_prime.cpu().numpy())
                                mat = torch.from_numpy(mat)
                                if is_GPU:
                                    mat = mat.to(device)
                            hessian_para_prod = compute_approx_hessian_vector_prod_with_prepared_terms1(zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat, vec_para_diff, is_GPU, device)
                        else:
                            hessian_para_prod, zero_mat_dim, curr_Y_k, curr_S_k, sigma_k, mat_prime = cal_approx_hessian_vec_prod0_3(S_k_list, Y_k_list, i, init_epochs, m, vec_para_diff, period, is_GPU, device)
        
                        exp_gradient, exp_param = None, None
                        if gradient_dual is not None:
                            is_positive, final_gradient_list = compute_grad_final3(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), get_all_vectorized_parameters1(gradient_dual), grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, is_GPU, device)
                        else:
                            is_positive, final_gradient_list = compute_grad_final3(get_all_vectorized_parameters1(para), torch.t(hessian_para_prod), None, grad_list_GPU_tensor[cached_id], para_list_GPU_tensor[cached_id], end_id - j, curr_matched_ids_size, learning_rate, regularization_coeff, is_GPU, device)
                        
                        vec_para = update_para_final2(para, final_gradient_list, learning_rate, regularization_coeff, exp_gradient, exp_param)
                        para = get_devectorized_parameters(vec_para, full_shape_list, shape_list)
                        
                i = i + 1
                j += batch_size
                cached_id += 1
                
                if cached_id%cached_size == 0:
                    GPU_tensor_end_id = (batch_id + 1)*cached_size
                    if GPU_tensor_end_id > para_list_all_epochs_tensor.shape[0]:
                        GPU_tensor_end_id = para_list_all_epochs_tensor.shape[0] 
                    
                    if (GPU_tensor_end_id/cached_size)%100 == 0:
                        print("end_tensor_id::", GPU_tensor_end_id)
                    
                    para_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(para_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                    grad_list_GPU_tensor[0:GPU_tensor_end_id - batch_id*cached_size].copy_(gradient_list_all_epochs_tensor[batch_id*cached_size:GPU_tensor_end_id])
                    batch_id += 1
                    cached_id = 0
                
                id_start = id_end
                            
                
        # print('overhead::', overhead)
        print(f'Class: {class_id} overhead2:: {overhead2}')
        print(f'Class: {class_id} overhead3:: {overhead3}')
        print(f'Class: {class_id} overhead4:: {overhead4}')
        print(f'Class: {class_id} overhead5:: {overhead5}')
                    
        init_model(model, para)
        updated_models.append(model)

    return updated_models


def model_update_del_ovr(args, method, lr_lists):

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

    X_remove = dataset_train.data[delta_data_ids]
    y_remove = dataset_train.labels[delta_data_ids]

    dataset_remove = dataset_train.__class__(X_remove, y_remove,ovr=True)
    remain_ids = torch.LongTensor(
        list(set(range(len(dataset_train))) - set(delta_data_ids))
    )

    X_prime = dataset_train.data[remain_ids]
    y_prime = dataset_train.labels[remain_ids]
    dataset_prime = dataset_train.__class__(X_prime, y_prime,ovr=True)

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
            compute_model_para_diff_ovr(origin_models,updated_models)
            test_ovr(
                updated_models,
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
                updated_models,
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
                updated_models,
                dataset_prime,
                batch_size,
                criterion,
                len(dataset_prime),
                is_GPU,
                device,
                num_class
            )

    else:
        if method == deltagrad_method:

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
                compute_model_para_diff_ovr(model_base_line, updated_models)
                test_ovr(
                    updated_models,
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
                    updated_models,
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
                    updated_models,
                    dataset_prime,
                    batch_size,
                    criterion,
                    len(dataset_prime),
                    is_GPU,
                    device,
                    num_class
                )


def generate_random_id_del(git_ignore_folder, dataset_train, epochs):

    #     delta_data_ids = torch.load(git_ignore_folder + 'delta_data_ids')
    #
    #     torch.save(dataset_train.data[delta_data_ids], git_ignore_folder + 'X_to_add')
    #
    #     torch.save(dataset_train.labels[delta_data_ids], git_ignore_folder + 'Y_to_add')
    #
    #
    #     selected_rows = get_subset_training_data(dataset_train.data.shape[0], delta_data_ids)
    #
    #     dataset_train.data = dataset_train.data[selected_rows]
    #
    #     dataset_train.labels = dataset_train.labels[selected_rows]

    generate_random_ids_list(dataset_train, epochs, git_ignore_folder)


def main_del_ovr(args, lr_lists):

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

    dataset_train = torch.load(git_ignore_folder + "dataset_train")

    dataset_test = torch.load(git_ignore_folder + "dataset_test")

    data_preparer = Data_preparer()

    dim = [len(dataset_train), len(dataset_train[0][0])]

    num_class = get_data_class_num_by_name(data_preparer, dataset_name)

    hyper_para_function = getattr(Data_preparer, "get_hyperparameters_" + dataset_name)
    # create a model for each class
    models = []
    noise_vectors_ovr = None
    criterion=None
    optimizer_ovr = []
    for class_id in range(num_class):
        if model_name == "LR":
            model = model_class(dim[1])
        else:
            model = model_class()
        if is_GPU:
            model.to(device)
        
        # create noise vectors for each model 
        if args.remove_method == "Guo":
            noise_vectors = []
            # set seed during the first class
            print("Creating Noise Vector")
            if class_id == 0:
                torch.manual_seed(args.noise_seed)
            for param in model.parameters():
                param_noise = torch.randn(param.size()) * args.noise_std
                noise_vectors.append(param_noise)
            
            noise_vectors_ovr.append(noise_vectors)

        models.append(model)
        
        criterion, optimizer = hyper_para_function(
            data_preparer, model.parameters(), lr_lists[0], regularization_coeff
        )
        optimizer_ovr.append(optimizer)

    hyper_params = [criterion, optimizer_ovr]

    generate_random_id_del(git_ignore_folder, dataset_train, num_epochs)

    random_ids_all_epochs = torch.load(git_ignore_folder + "random_ids_multi_epochs")

    t1 = time.time()
    # train the ovr models
    (
        models,
        gradient_list_all_epochs_ovr,
        para_list_all_epochs_ovr,
        learning_rate_all_epochs_ovr,
    ) = model_ovr_training_lr_test(
        random_ids_all_epochs,
        num_epochs,
        models,
        dataset_train,
        len(dataset_train),
        optimizer_ovr,
        criterion,
        batch_size,
        is_GPU,
        device,
        lr_lists,
        num_classes=num_class,
        noise_vectors_ovr = noise_vectors_ovr,
    )

    t2 = time.time()
    print("training time full::", t2 - t1)
    torch.save(dataset_train, git_ignore_folder + "dataset_train")
    torch.save(dataset_test, git_ignore_folder + "dataset_test")

    torch.save(gradient_list_all_epochs_ovr, git_ignore_folder + "gradient_list_all_epochs_ovr")

    torch.save(para_list_all_epochs_ovr, git_ignore_folder + "para_list_all_epochs_ovr")

    torch.save(learning_rate_all_epochs_ovr, git_ignore_folder + "learning_rate_all_epochs_ovr")

    torch.save(num_epochs, git_ignore_folder + "epoch")

    torch.save(hyper_params, git_ignore_folder + "hyper_params")

    torch.save(noise_vectors_ovr, git_ignore_folder + "noise_vectors_ovr")

    save_random_id_orders(git_ignore_folder, random_ids_all_epochs)

    # collect the init parameters for all class
    init_parameters_class = [para_list_all_epochs_ovr[class_id][0] for class_id in range(num_class)]
    torch.save(init_parameters_class, git_ignore_folder + "init_para_class")

    torch.save(models, git_ignore_folder + "origin_model_ovr")
    
    torch.save(model_class, git_ignore_folder + "model_class")

    torch.save(regularization_coeff, git_ignore_folder + "beta")

    torch.save(dataset_name, git_ignore_folder + "dataset_name")

    torch.save(batch_size, git_ignore_folder + "batch_size")

    torch.save(device, git_ignore_folder + "device")

    torch.save(is_GPU, git_ignore_folder + "is_GPU")

    # test the model 
    test_ovr(models, dataset_test, batch_size, criterion, len(dataset_test), is_GPU, device, num_classes=num_class)
