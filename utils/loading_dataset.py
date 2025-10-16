""" 
Author : Md Hasib Al Muzdadid Haque Himel
Email : muzdadid@gmail.com

Copyright (c) 2025, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""
 

import pandas as pd
from normalizer import normalize



def process_label(label): 
    # adjusts person name token labels  
    name_label=["B-PER", "I-PER"]
    new_name_label= ["B-NAME", "I-NAME"]
    labellist= []
    for i in label:
        if i in name_label:
            labellist.append(new_name_label[name_label.index(i)])
        else:
            labellist.append('O')      
    return labellist



def readfile(filename):
    # reads the dataset file and converts into a dataframe
    file = open(filename, encoding="utf-8")
    sentences = []
    sentence = []
    labels= []
    label= []
    for line in file:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n" or line[1]=="\n":
            if len(sentence) > 0:
                sentence= " ".join(sentence)
                sentences.append(sentence)
                labels.append(label)
                sentence = []
                label= []
            continue
        
        splits = line.split('\t')
        if (splits[-1]=='\n'):
            continue
        sentence.append(splits[0])
        label.append(splits[-1].split("\n")[0]) # remove extra "\n"

    if len(sentence) >0: # for last sentence
        sentence= " ".join(sentence)
        sentences.append(sentence)
        labels.append(label)
        sentence = []
        label= []
    
    file.close()

    df= pd.DataFrame(columns=["sentences", "labels"])
    df['sentences']= sentences
    df['labels']= labels
    df['labels']= df['labels'].apply(lambda x: process_label(x))
    df['sentences']= df['sentences'].apply(lambda x: normalize(x))
    
    return df

