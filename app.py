from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import io
import re
import emoji
import logging
import nltk
import matplotlib.pyplot as plt
from email.utils import parseaddr
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import PorterStemmer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from gensim.models import Word2Vec

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Logging setup
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load models
with open('word2vec_model.pkl', 'rb') as f:
    w2v = pickle.load(f)
with open('random_forest_model.pkl', 'rb') as model_file:
    random_forest_model = pickle.load(model_file)

# Load TLD frequency data
try:
    tld_data = pd.read_csv('tld_data_selected.csv')
except Exception as e:
    logging.error(f"Error loading TLD data: {e}")
    tld_data = pd.DataFrame(columns=['TLD', 'TLD_Freq'])

def stopwordslist():
    languages = ['english', 'french', 'german', 'spanish', 'italian']
    return set(word for lang in languages for word in stopwords.words(lang))

stopW = stopwordslist()

def tokenize_text(text):
    return word_tokenize(text.lower())

def clean_tokens(tokens):
    return [t for t in tokens if t.isalpha() or emoji.is_emoji(t)]

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stopW]

def lemmatize_tokens(tokens):
    ps = PorterStemmer()
    return [wn.morphy(t) or ps.stem(t) for t in tokens]

def extract_tld(email):
    _, address = parseaddr(email.replace('"', ''))
    domain = address.split('@')[-1] if '@' in address else ''
    return domain.split('.')[-1] if domain else ''

def get_vector(tokens, model):
    vectors = [model.wv[i] for i in tokens if i in model.wv.key_to_index]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def preprocess_data(data):
    attachment_count = int(data.get('Attachment Count', 0))
    email_subject = data.get('Email Subject', '')
    email_from = data.get('Email From', '')
    
    tld = extract_tld(email_from)
    tld_freq = tld_data[tld_data['TLD'] == tld]['TLD_Freq'].values[0] if tld in tld_data['TLD'].values else 0.0776
    
    tokens = tokenize_text(email_subject)
    tokens = clean_tokens(tokens)
    tokens = remove_stopwords(tokens)
    lemmatized_tokens = lemmatize_tokens(tokens)
    vector = get_vector(lemmatized_tokens, w2v)
    
    df_encoded = pd.DataFrame({
        'Attachment Count': [attachment_count],
        'Email_From_Length': [len(email_from)],
        'TLD_Freq': [tld_freq],
        **{f'embedding_{i}': [vector[i]] for i in range(len(vector))}
    })
    
    return df_encoded

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = preprocess_data(data['features'])
        predictions = random_forest_model.predict(df)
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict_prob', methods=['POST'])
def predict_probs():
    try:
        data = request.get_json()
        df = preprocess_data(data['features'])
        predictions_probs = random_forest_model.predict_proba(df)
        winning_class = int(predictions_probs.argmax(axis=1)[0])
        winning_prob = float(predictions_probs.max(axis=1)[0])
        return jsonify({'winning_class': winning_class, 'winning_prob': winning_prob})
    except Exception as e:
        logging.error(f"Prediction probability error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict_prob_from_csv', methods=['POST'])
def predict_prob_from_csv():
    try:
        file = request.files['file']
        df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))
        results = []
        for _, row in df.iterrows():
            features = row.to_dict()
            preprocessed_df = preprocess_data(features)
            predictions_probs = random_forest_model.predict_proba(preprocessed_df)
            results.append({
                'features': features,
                'winning_class': int(predictions_probs.argmax(axis=1)[0]),
                'winning_prob': float(predictions_probs.max(axis=1)[0])
            })
        return jsonify(results)
    except Exception as e:
        logging.error(f"CSV Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
