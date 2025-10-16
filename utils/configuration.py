""" 
Author : Md Hasib Al Muzdadid Haque Himel
Email : muzdadid@gmail.com

Copyright (c) 2025, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""
 
 
import torch

class CONFIG:

    train= True #False
    debug= False #True
    seed= 42
    output_dir= "./working/"
    train_data_path= "./dataset/dataset_train.txt"
    test_data_path= "./dataset/dataset_test.txt" 
  
    n_folds= 3
    num_epochs= 50
    label_names=["O", "B-NAME", "I-NAME"]
    num_labels= len(label_names)
    model_name= "celloscopeai/celloscope-28000-ner-banglabert-finetuned"  #"csebuetnlp/banglabert"  #"csebuetnlp/banglabert_large" 
    model_checkpoint= "./working/best_model_0.bin"
    max_length= 126
    
    do_normalize= True
    do_downsampling= False
    do_upsampling= False
    upsample_size= 1

    train_batch_size= 8
    valid_batch_size= 16
    test_batch_size= 16
    num_workers= 2
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    patience= 3
    gradient_accumulation_steps= 1
    learning_rate= 2e-5
    weight_decay= 1e-2
    scheduler= "CosineAnnealingWarmRestarts" #"CosineAnnealingLR" #"linear"
    T_max= 500
    T_0= 500
    min_lr= 1e-7
    eps = 1e-6
    betas= [0.9, 0.999]
    
    if debug:
        n_folds= 2
        num_epochs=2
        dataset_size= 300
