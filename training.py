""" 
Author : Md Hasib Al Muzdadid Haque Himel
Email : muzdadid@gmail.com

Copyright (c) 2025, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""


import pandas as pd
import numpy as np
import argparse
from normalizer import normalize # !pip install git+https://github.com/csebuetnlp/normalizer --quiet
from utils.configuration import CONFIG
from utils.loading_dataset import readfile
from utils.data_preprocessing import remove_erroneous_entries, upsampling, downsampling, train_validation_kfold
from utils.training_utils import get_tokenizer, Collate, prepare_loader, NER_MODEL, get_optimizer, fetch_scheduler, training_loop


def main():

    # parse the args
    parser= argparse.ArgumentParser(description= "Training for Bengali Name Extraction")
    parser.add_argument("--model_name", type= str, default= CONFIG.model_name, \
                        help="Provide huggingface language model for token classification", \
                        choices=["celloscopeai/celloscope-28000-ner-banglabert-finetuned", 
                                "csebuetnlp/banglabert",
                                "csebuetnlp/banglabert_large"])
    parser.add_argument("--do_normalize", type= bool, default= CONFIG.do_normalize, help= "Normalize input text or not")
    parser.add_argument("--do_downsampling", type= bool, default= CONFIG.do_downsampling, help= "Downsample majority class or not")
    parser.add_argument("--do_upsampling", type= bool, default= CONFIG.do_upsampling, help= "Upsample minority class or not")
    parser.add_argument("--upsample_size", type= float, default= CONFIG.upsample_size, help= "Percent of data to be upsampled")
    parser.add_argument("--output_dir", type= str, default= CONFIG.output_dir, help= "Path to the output files")
    parser.add_argument("--debug", type= bool, default= CONFIG.debug, help= "Debug or full trainig")
    parser.add_argument("--n_folds", type= int, default= CONFIG.n_folds, help= "Number of folds for training")
    parser.add_argument("--num_epochs", type= int, default= CONFIG.num_epochs, help= "Number of epochs for running")
    parser.add_argument("--train_batch_size", type= int, default= CONFIG.train_batch_size, help= "Training batch size")
    parser.add_argument("--valid_batch_size", type= int, default= CONFIG.valid_batch_size, help= "Validation batch size")
    parser.add_argument("--gradient_accumulation_steps", type= int, default= CONFIG.gradient_accumulation_steps, help= "Gradient accumulation steps")
    parser.add_argument("--learning_rate", type= float, default= CONFIG.learning_rate, help= "Initial learning rate")
    parser.add_argument("--scheduler", type= str, default= CONFIG.scheduler, help="Learning rate scheduler",
                        choices=["CosineAnnealingWarmRestarts", "CosineAnnealingLR", "linear"])
    parser.add_argument("--max_length", type= int, default= CONFIG.max_length, help= "Maximum sequence length") 
    
    args = parser.parse_args()
    CONFIG.model_name= args.model_name
    CONFIG.do_normalize= args.do_normalize
    CONFIG.do_downsampling= args.do_downsampling
    CONFIG.do_upsampling= args.do_upsampling
    CONFIG.upsample_size= args.upsample_size
    CONFIG.output_dir= args.output_dir
    CONFIG.debug= args.debug
    CONFIG.n_folds= args.n_folds
    CONFIG.num_epochs= args.num_epochs
    CONFIG.train_batch_size= args.train_batch_size
    CONFIG.valid_batch_size= args.valid_batch_size
    CONFIG.gradient_accumulation_steps= args.gradient_accumulation_steps
    CONFIG.learning_rate= args.learning_rate
    CONFIG.scheduler= args.scheduler
    CONFIG.max_length= args.max_length

    # load dataset
    train_dataset= readfile(file_name= CONFIG.train_data_path)
    dataset= remove_erroneous_entries(train_dataset)
 
    if CONFIG.do_upsampling:
        dataset= upsampling(df= dataset, upsample_size= 1.0)
    if CONFIG.do_downsampling:
        dataset= downsampling(df= dataset) 

    # small datset for debuging
    if CONFIG.debug:
        data= dataset[['sentences', 'labels']][: CONFIG.dataset_size]
    else:
        data= dataset[['sentences', 'labels']]
    
    # kfold data for training
    data = train_validation_kfold(data= data, n_folds= CONFIG.n_folds, seed= CONFIG.seed)

    # loading tokenizer and collate function
    tokenizer= get_tokenizer(model_name= CONFIG.model_name)
    collate_fn= Collate(tokenizer= tokenizer)
 
    fold_scores= []
    for fold in range(CONFIG.n_folds):
        torch.cuda.empty_cache()
        print(f"====== Started Training Fold: {fold} ======")

        train_loader, valid_loader= prepare_loader(df= data, tokenizer= tokenizer, fold= fold, collate_fn= collate_fn, cfg= CONFIG)
        model= NER_MODEL(cfg= CONFIG)
        model.to(device= CONFIG.device)
        optimizer= get_optimizer(model.parameters(), cfg= CONFIG)
        scheduler= fetch_scheduler(optimizer= optimizer, cfg= CONFIG)

        history, epoch_loss, f1_score= training_loop(model,  train_loader, valid_loader, optimizer, scheduler, fold= fold, cfg= CONFIG, num_epochs= CONFIG.num_epochs, patience= CONFIG.patience)
        print("\n\n")
        print(f"Fold [{fold}] avg loss: {epoch_loss}\n")
        print(f"Fold [{fold}] avg score: {f1_score}\n")
        fold_scores.append(f1_score)
        
        if fold < CONFIG.n_folds-1:
            del model
        del train_loader, valid_loader

    print(f"====== ====== ====== ======")
    print(f"Overall score: {np.mean(np.mean(fold_scores, axis= 0))}")
    print(f"====== ====== ====== ======")




if __name__ == "__main__":
    main()
 