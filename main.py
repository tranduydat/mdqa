
from model import model_drsbert

from sentence_transformers import SentenceTransformer, util
from pytorch_transformers import BertForQuestionAnswering, BertTokenizer

import os
from tqdm import tqdm

import pandas as pd
import time

dirname, filename = os.path.split(os.path.abspath(__file__))

if __name__== "__main__" : 

    sbert = SentenceTransformer('paraphrase-distilroberta-base-v1')
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    modelQA = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    model = model_drsbert.QuestionAnswering(sbert, tokenizer, modelQA, 30)

    st = time.time()
    model.saveParam(os.path.join(dirname,"data/cea201.txt"),os.path.join(dirname,"encoded/cea201.pkl"))
    ed = time.time()
    print("Encoded Time: {}".format(ed-st))

    model.loadParam(os.path.join(dirname,"encoded/cea201.pkl"))

    df = pd.read_csv(os.path.join(dirname,"data/cea201_quizzlet.csv"))

    save_df = {"question":[],"A":[],"B":[],"C":[],"D":[],"true_answer":[],"predict_answer":[],"predict_data":[],"tag":[]}

    answer_box = ["A","B","C","D"]
    acc = 0
    for idx in range(len(df["question"])):
        
        question = df["question"][idx]
        ex_answer = [df["A"][idx],df["B"][idx],df["C"][idx],df["D"][idx]]
        right_answer = df["answer"][idx]
        model_answer, ans, sc = model.FindAnswer(question,ex_answer)

        save_df["question"].append(df["question"][idx])
        save_df["A"].append(df["A"][idx])
        save_df["B"].append(df["B"][idx])
        save_df["C"].append(df["C"][idx])
        save_df["D"].append(df["D"][idx])
        save_df["true_answer"].append(df["answer"][idx])
        save_df["predict_answer"].append(answer_box[model_answer])
        save_df["predict_data"].append(ans)    

        print("\n****Question {}****".format(question))
        print("A. {}\nB. {}\nC. {}\n D. {}\n".format(df["A"][idx],df["B"][idx],df["C"][idx],df["D"][idx]))
        print("Model answer: {} - Right Answer: {}.".format(ex_answer[model_answer], df[answer_box[model_answer]][idx]))

        if model_answer != -1 and (answer_box[model_answer] in right_answer):
            acc += 1
            save_df["tag"].append(1)
        else:
            save_df["tag"].append(0)
            
    print("Accuracy: {}/{} - {}".format(acc,len(df["question"]),acc/len(df["question"])))

    pd.DataFrame(save_df).to_csv(os.path.join(dirname,"result/cea201_quizzlet_result.csv"))