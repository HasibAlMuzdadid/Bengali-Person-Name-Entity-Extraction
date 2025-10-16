## Project : Bengali Person Name Entity Extraction
[![Author](https://img.shields.io/badge/Author-Hasib%20Al%20Muzdadid-brightgreen)](https://github.com/HasibAlMuzdadid)
[![BSD 3-Clause License](https://img.shields.io/github/license/hasibalmuzdadid/Bengali-Person-Name-Entity-Extraction?style=flat&color=orange)](https://github.com/HasibAlMuzdadid/Bengali-Person-Name-Entity-Extraction/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/hasibalmuzdadid/Bengali-Person-Name-Entity-Extraction?style=social)](https://github.com/HasibAlMuzdadid/Bengali-Person-Name-Entity-Extraction/stargazers)

**Author :** </br>
Hasib Al Muzdadid</br>
[Department of Computer Science & Engineering](https://www.cse.ruet.ac.bd/), </br>
[Rajshahi University of Engineering & Technology (RUET)](https://www.ruet.ac.bd/) </br>
Portfolio: https://hasibalmuzdadid.github.io  </br> 

 
## Project Description :
Built an intelligent Bengali person name entity extraction engine that reads bengali texts and instantly detects who is mentioned or smartly detects when no one is. 

**For Example :**
<br>-->input : শিল্প মন্ত্রণালয়ের সচিব মো. আব্দুর রহিম বলেন, পবিত্র রমজান মাস এলেই ব্যবসায়ীদের মধ্যে বেশি মুনাফা করার প্রবণতা তৈরি হয়
<br>-->output : মো. আব্দুর রহিম


**Language used :** Python  </br> 
**ML framework :** PyTorch  </br>
**Dataset used :** <a href="https://github.com/Rifat1493/Bengali-NER/tree/master/Input">[Bengali NER]</a>  </br>
**Models used :** </br> 
1. <a href= "https://huggingface.co/celloscopeai/celloscope-28000-ner-banglabert-finetuned">ner-banglabert-finetuned</a>
2. <a href= "https://huggingface.co/csebuetnlp/banglabert">banglabert</a>
3. <a href= "https://huggingface.co/csebuetnlp/banglabert_large">banglabert-large</a>  </br>

**Results :** </br>
| **Model**                                                                   | **F1 Score** |
|:----------------------------------------------------------:|:--------------------------------
| banglabert                                                                  | 0.8491          |
| banglabert + upsampled                                                      | 0.8289          |
| banglabert + downsampled                                                    | 0.8515          |
| **ner-banglabert-finetuned**                                                | **0.8518**      |
| ner-banglabert-finetuned + upsampled                                        | 0.8369          |
| ner-banglabert-finetuned + downsampled                                      | 0.8512          |
| banglabert-large                                                            | 0.8427          |
| banglabert-large + upsampled                                                | 0.8414          |
| banglabert-large + downsampled                                              | 0.8359          |


 
## Setup :
For installing the necessary requirements, use the following bash snippet
```bash
git clone https://github.com/HasibAlMuzdadid/Bengali-Person-Name-Entity-Extraction.git
cd Bengali-Person-Name-Entity-Extraction/
python -m venv myenv
myenv/Scripts/Activate 
pip install -r requirements.txt
```
N.B: Modify the commands appropriately based on the terminal you are using.

