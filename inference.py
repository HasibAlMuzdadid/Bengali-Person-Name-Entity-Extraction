""" 
Author : Md Hasib Al Muzdadid Haque Himel
Email : muzdadid@gmail.com

Copyright (c) 2025, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""


import argparse
import torch
from utils.configuration import CONFIG
from normalizer import normalize
from utils.training_utils import get_tokenizer, NER_MODEL
from utils.inference_utils import prediction, show_names


def main():

    # parse the args
    parser= argparse.ArgumentParser(description= "Inference for Bengali Name Extraction")
    parser.add_argument("--text", help= "Text for extracting name (list of texts is preferable)",
                        default= "শিক্ষা মন্ত্রণালয়ের দায়িত্বশীল একটি সূত্রে জানা যায়, মুহাম্মদ আজাদ খানের বিষয়ে শিক্ষা মন্ত্রণালয়ের নীতিনির্ধারকেরা সন্তুষ্ট ছিলেন না।")
    parser.add_argument("--model_name", type= str, default= CONFIG.model_name, \
                        help="Provide huggingface language model for token classification", \
                        choices=["celloscopeai/celloscope-28000-ner-banglabert-finetuned", 
                                "csebuetnlp/banglabert",
                                "csebuetnlp/banglabert_large"])
    parser.add_argument("--model_checkpoint", type= str, default= CONFIG.model_checkpoint, help= "Path of the saved model")
    
    
    args = parser.parse_args()
    text= args.text
    CONFIG.model_name= args.model_name
    CONFIG.model_checkpoint= args.model_checkpoint

    tokenizer = get_tokenizer(model_name= CONFIG.model_name)
    model= NER_MODEL(cfg= CONFIG)
    model.load_state_dict(torch.load(CONFIG.model_checkpoint, map_location=  CONFIG.device))
    model.to(CONFIG.device)

    if text != None:
        outputs=[]
        if type(text) == str:
            text= normalize(text)
            output= prediction(text, model, tokenizer, CONFIG)
            outputs.append(output)
        elif type(text)== list:
            for txt in text:
                txt= normalize(txt)
                output= prediction(txt, model, tokenizer, CONFIG)
                outputs.append(output)
        else:
            outputs= None
            print("Please give input in string format or list of strings")

        if outputs != None:
            show_names(text, outputs, tokenizer)
    else:
        print("Please give text input in proper format")



if __name__ == "__main__":
    main()


"""
Please give text input as one of the following format.
text= "শিক্ষা মন্ত্রণালয়ের দায়িত্বশীল একটি সূত্রে জানা যায়, মুহাম্মদ আজাদ খানের বিষয়ে শিক্ষা মন্ত্রণালয়ের নীতিনির্ধারকেরা সন্তুষ্ট ছিলেন না।"
or 
text= ["অভিনেত্রী মন্দিরা চক্রবর্তীকে পূজার চার সাজে সাজানোর সময় ডিজাইনার হিসেবে আরাম, স্বকীয়তা আর স্বাচ্ছন্দ্যকে প্রাধান্য দিয়েছি বেশি। প্রতিটি সাজেই ছিল টেকসই নকশার ছোঁয়া।",
       "শিল্প মন্ত্রণালয়ের সচিব মো. আব্দুর রহিম বলেন, পবিত্র রমজান মাস এলেই ব্যবসায়ীদের মধ্যে বেশি মুনাফা করার প্রবণতা তৈরি হয়।",
       "শিক্ষা মন্ত্রণালয়ের দায়িত্বশীল একটি সূত্রে জানা যায়, মুহাম্মদ আজাদ খানের বিষয়ে শিক্ষা মন্ত্রণালয়ের নীতিনির্ধারকেরা সন্তুষ্ট ছিলেন না।"
      ]
"""
