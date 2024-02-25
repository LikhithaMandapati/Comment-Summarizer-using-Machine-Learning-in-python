from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import T5Generator
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from transformers import BartForConditionalGeneration, BartTokenizer


app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("allenai/tk-instruct-base-def-pos")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/tk-instruct-base-def-pos")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None

    if request.method == 'POST':
        text = request.form['text']
        print(text)
        choice = request.form['choice']

        if choice == 'Detect Aspects':
            prediction_result = predict_aspects(text)
        elif choice == 'Detect Sentiment':
            prediction_result = predict_sentiment(text)
        elif choice == 'Summarization':
            prediction_result = predict_sentiment_custom(text)

    return render_template('index.html', prediction_result=prediction_result)

@app.route('/predict', methods=['POST'])
def predict_aspects(text):
    model_out_path = 'model_directory'
    t5_exp = T5Generator(model_out_path)
    model_input = text
    input_ids = t5_exp.tokenizer(model_input, return_tensors="pt").input_ids
    outputs = t5_exp.model.generate(input_ids, max_length=50)
    predicted_aspects = t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('Predicted Aspects:', predicted_aspects)
    return predicted_aspects

def predict_sentiment(text):
    tokenizer = AutoTokenizer.from_pretrained("kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-combined")
    model = AutoModelForSeq2SeqLM.from_pretrained("kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-combined")

    bos_instruction = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
    Positive example 1-
    input: I charge it at night and skip taking the cord with me because of the good battery life.
    output: battery life:positive,
    Positive example 2-
    input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
    output: features:positive, iChat:positive, Photobooth:positive, garage band:positive
    Negative example 1-
    input: Speaking of the browser, it too has problems.
    output: browser:negative
    Negative example 2-
    input: The keyboard is too slick.
    output: keyboard:negative
    Neutral example 1-
    input: I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset.
    output: battery:neutral
    Neutral example 2-
    input: Nightly my computer defrags itself and runs a virus scan.
    output: virus scan:neutral
    Now complete the following example-
    input: """
    delim_instruct = ''
    eos_instruct = ' \noutput:'

    tokenized_text = tokenizer(bos_instruction + text + delim_instruct + eos_instruct, return_tensors="pt")
    output = model.generate(tokenized_text.input_ids)
    sentiment = tokenizer.decode(output[0], skip_special_tokens=True)
    print('Model output: ', tokenizer.decode(output[0], skip_special_tokens=True))
    return sentiment

def predict_sentiment_custom(text):
    extractive_summary = text_rank_summary(text)
    abstractive_summary = bart_summarize(extractive_summary)
    return [abstractive_summary,extractive_summary]


def text_rank_summary(document, max_sentences=3):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(document)
    tokens = [token.text for token in doc]
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in STOP_WORDS:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/max_frequency)

    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    summarized_sentences = nlargest(max_sentences, sentence_scores, key=sentence_scores.get)
    final_sentences = [w.text for w in summarized_sentences]
    summary = ' '.join(final_sentences)
    return summary


def bart_summarize(text):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == '__main__':
    import gdown
    import zipfile
    import os
    url = 'https://drive.google.com/uc?id=1jDFM2_JLyaHK2l-pkp7CwOx2ouU7_qJ9'
    output_zip = 'allenaitk-instruct-base-def-pos-aspects.zip'
    gdown.download(url, output_zip, quiet=False)
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall('model_directory')
    os.remove(output_zip)
    app.run(debug=True)

