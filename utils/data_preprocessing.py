""" 
Author : Md Hasib Al Muzdadid Haque Himel
Email : muzdadid@gmail.com

Copyright (c) 2025, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""


import random
import pandas as pd
from bnlp import BasicTokenizer
from sklearn.model_selection import StratifiedKFold



def check_common_entry(df1, df2):
    # checks wheather the train and test datasets have common entries
    b_tokenizer= BasicTokenizer()
    list1= [tuple(b_tokenizer.tokenize(txt)) for txt in df1['sentences']]
    list2= [tuple(b_tokenizer.tokenize(txt)) for txt in df2['sentences']]
    common= set(list1).intersection(set(list2))
    return common



def remove_common_entries(df, common):
    # removes common entries of the train and test datasets
    b_tokenizer= BasicTokenizer()
    df_tokens = df['sentences'].apply(lambda x: tuple(b_tokenizer.tokenize(x)))  # pre-tokenize all df sentences
    common_set = set(common)  # convert common to set for fast lookup
    mask = df_tokens.apply(lambda x: x not in common_set)  # mask for rows to keep (sentence tokens not in common_set)
    df_filtered = df[mask].reset_index(drop=True)
    
    removed_count = len(df) - len(df_filtered)
    print(f"Total {removed_count} rows removed from the dataset.")
    return df_filtered



def remove_erroneous_entries(df):
    # removes erroneous entries which have unequal no of labels or words
    temp_df= df.copy()
    b_tokenizer= BasicTokenizer()
    temp_df['len_labels']= temp_df['labels'].apply(lambda x: len(x))
    temp_df['len_words']= temp_df['sentences'].apply(lambda x: len(b_tokenizer.tokenize(x)))
    
    error_=[]
    for i in range(len(temp_df)):
        if temp_df['len_labels'][i] != temp_df['len_words'][i]:
            error_.append(i)
    print(f"{len(error_)} no of data was detected as erroneous and discarded.")
    df= df.drop(error_).reset_index(drop= True)
    return df



def downsampling(df):
    # down-samples majority class data
    random.seed(42)
    index_0= df[df['name_tag']==0].index   #indexes without name entity
    index_1= df[df['name_tag']==1].index   #indexes with name entity
    index_n= None
    if len(index_0) > len(index_1):
        index = [i for i in index_0]
        index_n= random.sample(index, k= len(index_0) - len(index_1))
    if index_n is not None:
        df= df.drop(index_n).reset_index(drop= True)
    return df



def upsampling(df, upsample_size=1.0):
    # upsamples minority class data
    random.seed(42)
    df['name_tag'] = df['labels'].apply(lambda x: 1 if x.count("B-NAME") > 0 else 0)
    index_0 = df[df['name_tag'] == 0].index
    index_1 = df[df['name_tag'] == 1].index

    if len(index_0) > len(index_1):
        n_diff = len(index_0) - len(index_1)
        k = int(n_diff * upsample_size)
        index_add = random.choices(index_1, k=k)
    elif len(index_1) > len(index_0):
        n_diff = len(index_1) - len(index_0)
        k = int(n_diff * upsample_size)
        index_add = random.choices(index_0, k=k)
    else:
        index_add = []

    if index_add:
        df = pd.concat([df, df.loc[index_add]]).reset_index(drop=True)
    
    return df



def train_validation_kfold(data, n_folds= 3, seed= 42):
    # splits data for training and validation
    data['name_tag']= data['labels'].apply(lambda x: 1 if x.count("B-NAME")>0 else 0)
    skf= StratifiedKFold(n_splits= n_folds, random_state= seed, shuffle= True)
    for fold, (train_index, val_index) in enumerate(skf.split(X= data, y= data['name_tag'])):
        data.loc[val_index, 'fold']= int(fold)
        
    data['fold']= data['fold'].astype(int)
    data= data[['sentences', 'labels', 'fold']]
    return data
