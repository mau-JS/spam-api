from flask import Flask, request, jsonify
import pickle
import category_encoders as ce
import numpy as np
import pandas as pd
from email.utils import parseaddr
import re
import emoji
from wordcloud import WordCloud
from nltk.corpus import PlaintextCorpusReader, stopwords, wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from gensim.models import word2vec
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


app = Flask(__name__)

with open('word2vec_model.pkl', 'rb') as f:
    w2v = pickle.load(f)


# Load the TLD_Freq data from the CSV file
tld_data = pd.read_csv('tld_data_selected.csv')

# Load the trained binary encoder (saved during training)
with open('binary_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)
    
###################################################
#Natural Language Preprocessing functions
def stopwordslist(languages):
    stop_list = []
    for i in languages:
        stop_list.extend(stopwords.words(i))
        #print("stop words",i)
    return stop_list
    
def tokenize_text(text):
    return word_tokenize(text)

def frequency_distance(tokens):
    fdist = nltk.FreqDist(tokens)
    return fdist

def frequency_tokens(tokens, fdist):
    dictionary = dict(fdist)
    frequency_df = pd.DataFrame.from_dict(
        dictionary,
        orient='index',
        columns=['Frequency']
    )
    frequency_df = frequency_df.sort_values(by=['Frequency'], ascending=False)
    frequency_df.index_name = 'Word'
    return frequency_df
    
def bigram_relation(fdist, tokens):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    bigram_fd = nltk.FreqDist(nltk.bigrams(tokens))
    finder = BigramCollocationFinder(fdist,bigram_fd)
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    bigram_df = pd.DataFrame(scored, columns=['Bigram','Score'])
    bigram_df.sort_values(by='Score',ascending=False)
    return bigram_df
    
def word_cloud(df):
    if df.empty:
        print("No valid data for word cloud.")
        return
    text_dict = df.to_dict()['Frequency']
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(text_dict)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def cleaning(tokens):
    clean_list = []
    for i in tokens:
        if i.isalpha():
            clean_list.append(i.lower())
        elif emoji.is_emoji(i):
            clean_list.append(i)
    return clean_list

def delete_stopwords(cleaned, stop_words):
    filtered_words = [token for token in cleaned if token not in stop_words]
    return filtered_words
    
def lemmatize(tokens):
    ps = PorterStemmer()
    lemmatize_stem = [wn.morphy(token.lower()) or ps.stem(token.lower()) for token in tokens]
    return lemmatize_stem

def get_vector(tokens,model):
    vectors = []
    for i in tokens:
        if i in model.wv:
            vectors.append(model.wv[i])
    if vectors:
        return sum(vectors) / len(vectors)
    else:
        return [0] * model.vector_size

def apply_get_vector(tokens_list, model):
    vector_list = []
    for i in tokens_list:
        vector = get_vector(i, model)
        vector_list.append(vector)
    return vector_list

###################################################
def preprocess_data(data):
    # Extract features from the data dictionary
    attachment_count = data['Attachment Count']
    attachment_extensions = data['Attachment Extension']  # Keep it as a raw string
    email_subject = data['Email Subject']

        # Convert 'Attachment Count' to integer (handle non-numeric values)
    try:
        attachment_count = int(attachment_count)
    except ValueError:
        attachment_count = 0  # Set to 0 if not a valid integer

    # Extract TLD from email addresses
    def extract_tld(email):
        email = email.replace('"', '')  # Remove double quotes
        _, address = parseaddr(email)
        domain = address.split('@')[-1] if '@' in address else ''
        tld = domain.split('.')[-1] if domain else ''
        return tld

    email_from = data.get('Email From', '')  # Handle missing 'Email From' gracefully
    tld = extract_tld(email_from)

    # Get the TLD_Freq corresponding to the extracted TLD
    tld_freq = tld_data[tld_data['TLD'] == tld]['TLD_Freq'].values[0]

    # Calculate the length of the email address
    email_from_length = len(email_from)
    spam = 'Spam'
    df_api = pd.DataFrame({
        'Attachment Count': [attachment_count],
        'Email Subject': [email_subject],
        'closeNotes':[spam],
        'Email_From_Length': [email_from_length],
        'TLD_Freq': [tld_freq]
    })

    if attachment_count > 0 and attachment_extensions != "":
        df_new = pd.DataFrame(np.repeat(df_api.values, attachment_count, axis=0))
        df_new.columns = df_api.columns

    elif attachment_count > 0 and attachment_extensions == "":
        df_new = df_api
    else:
        df_new = df_api
    


    df_extension = pd.DataFrame({"Attachment Extension": attachment_extensions.split(",")})
    # Create a DataFrame with a single row for the 'Attachment Extension'
    merged_df = pd.concat([df_new, df_extension], axis=1)
    # Display the merged DataFrame
    print(merged_df)
    # Apply binary encoding to the 'Attachment Extension'
    df_encoded = encoder.transform(merged_df)
    df_encoded = df_encoded[['Attachment Count','Attachment Extension_0','Attachment Extension_1','Attachment Extension_2','Attachment Extension_3','Attachment Extension_4',
                             'Attachment Extension_5','Attachment Extension_6','Attachment Extension_7','Email Subject','Email_From_Length','TLD_Freq']]
    df = df_encoded


    print(df_encoded)
####################################################
    #Natural Language Preprocessing for Email Subject
    corpus_subject = df_encoded['Email Subject']
    languages = ['arabic', 'azerbaijani', 'basque', 'bengali', 'catalan', 'chinese', 'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'greek',
                 'hebrew', 'hinglish', 'hungarian', 'indonesian', 'italian', 'kazakh', 'nepali', 'norwegian', 'portuguese', 'romanian', 'russian', 'slovene',
                 'spanish', 'swedish', 'tajik', 'turkish']
    stopW = stopwordslist(languages)
    corpus_subject = corpus_subject.astype(str)
    tokens = corpus_subject.apply(tokenize_text)
    #print(tokens)
    
    df_frequency= []
    frequency_original = []
    for i in tokens:
        frequency_original = frequency_distance(i)
        frequency = frequency_tokens(i,frequency_original)
        df_frequency.append(frequency)
        
    bigrams = []
    for i in tokens:
        frequency_original = frequency_distance(i)
        bigram_get = bigram_relation(frequency_original, i)
        bigrams.append(bigram_get)
        
#####Corpus Processing
    cleaned = []
    for i in tokens:
        cleaned.append(cleaning(i))
        
    filtered_cleaned = []
    for i in cleaned:
        filtered_cleaned.append(delete_stopwords(i,stopW))

    lematized = []
    for i in filtered_cleaned:
        lematized.append(lemmatize(i))

    frequency_cleaned = []
    df_frequency_cleaned = []
    for i in lematized:
        frequency_cleaned = frequency_distance(i)
        frequency = frequency_tokens(i, frequency_cleaned)
        df_frequency_cleaned.append(frequency)

    bigrams_cleaned = []
    for i in lematized:
        frequency_cleaned = frequency_distance(i)
        bigram_get = bigram_relation(frequency_cleaned,i)
        bigrams_cleaned.append(bigram_get)

    print(lematized)
    #####Corpus Extraction
    size_vector = 100
    context_max = 35
    min_presence = 1
    epochs = 50

    unique_token = []
    for i in lematized:
        unique_token.extend(i)

    #print("Number of unique words in the input data: ",len(unique_token))

    #w2v = word2vec.Word2Vec(sentences=[unique_token],
     #                   vector_size=size_vector,
      #                  window=context_max,
       #                 min_count=min_presence,
        #                epochs=epochs,
         #               sg=1
          #             )
    vectors = apply_get_vector(lematized, w2v)

    lematized_df = pd.DataFrame({'lematized': lematized})
    corpus_subject.reset_index(drop=True, inplace=True)
    corpus_subject = pd.concat([corpus_subject, lematized_df], axis=1)
    #print(corpus_subject)
    
    
    if(len(vectors) > len(lematized)):
        vectors = vectors[:1]
    corpus_subject['vectors'] = vectors
    #print(len(lematized))
    #print(len(vectors))
    #print(corpus_subject)

    df_encoded = df.join(corpus_subject['vectors'].set_axis(df.index))
    #print(df_encoded)

    pd.set_option('display.max_columns', None)
    # Replace 'df' with your DataFrame name
    df_encoded = df_encoded.drop('Email Subject', axis = 1)
    print(df_encoded)

# Return the preprocessed features as a dictionary
    return {
    'Attachment Count': attachment_count,
    'Email Subject': email_subject, # Assuming 'spam' is a predefined value
    'Email_From_Length': email_from_length,
    'TLD_Freq': tld_freq
    }




@app.route('/preprocess', methods=['POST'])  # Changed route to /preprocess
def preprocess():
    try:
        # Get data from POST request
        data = request.get_json()

        # Preprocess the data
        preprocessed_features = preprocess_data(data['features'])

        # Return preprocessed features (no prediction)
        return jsonify({'preprocessed_features': preprocessed_features})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)
