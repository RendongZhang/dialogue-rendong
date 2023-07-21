#import logging
import nltk
import os.path 
import sys
import io
from flask import Flask, request
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import pickle
import os
from sklearn.metrics import classification_report, accuracy_score, f1_score, plot_confusion_matrix,ConfusionMatrixDisplay, confusion_matrix
import pickle
os.environ["WANDB_DISABLED"] = "true"

labels_dict = {'y': 'yes-answers',
               'y-d': 'yes-answers-explanations',
               'n': 'no-answers',
               'n-d': 'no-answers-explanations',
               'sno': 'statements-non-opinion',
               'so': 'statements-opinion',
               'ack': 'acknowledgments',
               'dno': "other answers",
               'query': 'interviewee-initiated questions',
               'ft': 'thanking',
               'fa': 'apologies', 
               'fe': 'explanations',
               'fp': 'conventional'
              }

#Loading the Label Encoder 
le = pickle.load(open("Data/label_encoder.pkl", "rb"))

max_length=128
fine_tuned_model_name = 'Models/Final_model/BERT_13classes_final'

#Loading the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(fine_tuned_model_name, do_lower_case=True)

target_names = le.transform(le.classes_)

load_model = BertForSequenceClassification.from_pretrained(fine_tuned_model_name, num_labels=13)
load_model.to('cuda')

def preprocess_function(examples):
    global tokenizer
    #print(examples['TEXT'])
    return tokenizer(examples["text"], padding=True, truncation=True)
    #return toknizer(examples["text"],padding='max_length',truncation=True, max_length=max_length)


def get_new_prediction(text):
    global tokenizer, load_model
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to("cuda")
   # print(inputs)
    # perform inference to our model
    outputs = load_model(**inputs)
    #print(outputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1).argmax()
    #print(probs.item())
    #print(outputs[0].softmax(1).argmax())
    # executing argmax function to get the candidate label
    return probs.item()
    #return probs.argmax().item()


@app.route("/get_dialogue_act", methods=["PUT", "POST", "GET"])
def get_dialogue_act():
    return labels_dict[le.inverse_transform([get_new_prediction(text)])[0]]
    

if __name__ == '__main__':
    app.run(debug = True)

