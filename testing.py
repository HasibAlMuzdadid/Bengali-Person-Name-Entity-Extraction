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
import torch
from torch.utils.data import DataLoader
from utils.configuration import CONFIG
from utils.loading_dataset import readfile
from utils.data_preprocessing import check_common_entry, remove_common_entries, remove_erroneous_entries
from utils.training_utils import get_tokenizer, CustomDataset, Collate, NER_MODEL, testing_loop


def main():

    # parse the args
    parser= argparse.ArgumentParser(description= "Testing for Bengali Name Extraction")
    parser.add_argument("--test_data_path", type= str, default= CONFIG.test_data_path, help= "Path of the test data")
    parser.add_argument("--model_name", type= str, default= CONFIG.model_name, \
                        help="Provide huggingface language model for token classification", \
                        choices=["celloscopeai/celloscope-28000-ner-banglabert-finetuned", 
                                "csebuetnlp/banglabert",
                                "csebuetnlp/banglabert_large"])
    parser.add_argument("--do_normalize", type= bool, default= CONFIG.do_normalize, help= "Normalize input text or not")
    parser.add_argument("--model_checkpoint", type= str, default= CONFIG.model_checkpoint, help= "Path of the saved model")
    parser.add_argument("--test_batch_size", type= int, default= CONFIG.test_batch_size, help= "Testing batch size")
    parser.add_argument("--max_length", type= int, default= CONFIG.max_length, help= "Maximum sequence length")
    
    args = parser.parse_args()
    CONFIG.test_data_path= args.test_data_path
    CONFIG.model_name= args.model_name
    CONFIG.do_normalize= args.do_normalize
    CONFIG.model_checkpoint= args.model_checkpoint
    CONFIG.test_batch_size= args.test_batch_size
    CONFIG.max_length= args.max_length

    # load dataset
    train_dataset= readfile(file_name= CONFIG.train_data_path)
    test_dataset= readfile(file_name= CONFIG.test_data_path)

    common= check_common_entry(train_dataset, test_dataset) 
    test_data= remove_common_entries(test_dataset, common)
    test_data= remove_erroneous_entries(test_data)

    tokenizer= get_tokenizer(model_name= CONFIG.model_name)
    collate_fn= Collate(tokenizer= tokenizer)
    dataset= CustomDataset(df= test_data, tokenizer= tokenizer, cfg= CONFIG)
    test_loader= DataLoader(dataset= dataset,
                            batch_size= CONFIG.test_batch_size,
                            collate_fn= collate_fn,
                            num_workers= CONFIG.num_workers,
                            shuffle= False,
                            pin_memory= True,
                            drop_last= False
                            )
    
    model= NER_MODEL(cfg= CONFIG)
    model.load_state_dict(torch.load(CONFIG.model_checkpoint, map_location= CONFIG.device))
    model.to(CONFIG.device)

    print(f"Running test for model {CONFIG.model_name}")
    f1_score= testing_loop(model= model, dataloader= test_loader, device= CONFIG.device)
    print(f"Model {CONFIG.model_name} testing f1_score: {f1_score}")
    print("Finished Testing")


if __name__ == "__main__":
    main()

