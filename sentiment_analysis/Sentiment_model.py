
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import numpy as np



model_name = "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

dic={'MMM':0,'AXP':0,'AMGN':0,'AAPL':0,'BA':0,'CAT':0,'CVX':0,'CSCO':0,'KO':0,'DIS':0,'DOW':0,'GS':0,'HD':0,'HON':0,'IBM':0,'INTC':0,'JNJ':0,'JPM':0,'MCD':0,'MRK':0,'MSFT':0,'NKE':0,'PG':0,'CRM':0,'TRV':0,'UNH':0,'VZ':0,'V':0,'WBA':0,'WMT':0}




def get_sentiment_score(sentence, stock):
    out= classifier(sentence)
   # print(out)
    pos=0
    neg=0
    neutral=0
    sentiment_score=0
    for i in out:
       # print(i['label'])
        if(i['label']=='POSITIVE'):
            pos=i['score']
       #     print(pos)
        elif(i['label']=='NEGATIVE'):
            neg=i['score']
       #     print(neg)
        else:
            neutral= i['score']
            
    if(pos!=0 or neg!=0):
        sentiment_score= pos-neg
    else:
        sentiment_score=neutral

    if(dic[stock]==0):
        avg= sentiment_score
        dic[stock]=avg
        
    else:
        alpha = (calc_alpha(10,0.9))
        avg= dic[stock]
        res = update_ewma(avg,sentiment_score,alpha)
        dic[stock]=res
    
    return dic
    
def calc_alpha(window, weight_proportion):
    
    return 1 - np.exp(np.log(1-weight_proportion)/window)


def update_ewma(prev_stat, data_point, alpha):
    
    return data_point*alpha + (1-alpha) * prev_stat



