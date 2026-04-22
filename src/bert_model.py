from transformers import pipeline

bert_classifier = pipeline("sentiment-analysis")

def bert_predict(text):
    return bert_classifier(text)