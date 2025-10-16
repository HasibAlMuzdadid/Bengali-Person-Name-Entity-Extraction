""" 
Author : Md Hasib Al Muzdadid Haque Himel
Email : muzdadid@gmail.com

Copyright (c) 2025, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""


from utils.configuration import CONFIG



def prediction(text, model, tokenizer, cfg= CONFIG):
    # generates prediction for given text
    inputs= tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors="pt")
    outputs= model(inputs['input_ids'].to(cfg.device), inputs['attention_mask'].to(cfg.device))
    outputs= outputs.detach().cpu().numpy().argmax(axis= -1)[0, 1:-1]
    return outputs



def extract_spans(prediction):
    span_indices = [i for i, v in enumerate(prediction) if v != 0 ]
    span_list= []
    span= []
    
    for i in range(len(span_indices)):
        if i == 0 or span_indices[i] != span_indices[i-1]+1:
            if span:
                span_list.append(span)

            span= [span_indices[i]]

        else:
            span.append(span_indices[i])
    if span:
        span_list.append(span)
    
    return span_list



def extract_names(text, span_list, tokenizer):
    # extracts names from the given text and span list
    name_list= []
    if len(span_list) > 0:
        for span in span_list:
            tokens= tokenizer(text)['input_ids'][1:-1][span[0]:span[-1]+1]
            name= normalize(tokenizer.decode(tokens))
            name_list.append(name)
        return name_list
    else:
        return None


 
def show_names(texts, predictions, tokenizer):
    if type(texts)== str:
        texts= [texts]
    for text, pred in zip(texts, predictions):
        span_list= extract_spans(pred)
        name_list= extract_names(text, span_list, tokenizer)
        print(f"Given Text: {text} \nExtracted Names: {name_list}")