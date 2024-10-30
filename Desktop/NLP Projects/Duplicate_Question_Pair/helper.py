from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
import distance
import nltk
import string
import re
from textblob import TextBlob
import pickle
import pandas as pd

# Load the XGBoost model results
with open('C:\\Users\\91991\\Desktop\\NLP Projects\\Duplicate_Question_Pair\\results_xgb.pkl', 'rb') as f:
    results_xgb = pickle.load(f)

with open('C:\\Users\\91991\\Desktop\\NLP Projects\\Duplicate_Question_Pair\\tfidf_vectorizer.pkl', 'rb') as f:
    tfv = pickle.load(f)

nltk.download('stopwords')
def cnt_chr(text):  
    # Function to count characters  
    return len(text) if isinstance(text, str) else 0  

def common_words(q1, q2):  
    words_q1 = set(q1.lower().split())  
    words_q2 = set(q2.lower().split())  
    return len(words_q1.intersection(words_q2))
    
def basic_features(df):   
    df['question1'] = df['question1'].fillna('unknown').astype(str)  
    df['question2'] = df['question2'].fillna('unknown').astype(str)   
    df['len_q1'] = df['question1'].str.len()  
    df['len_q2'] = df['question2'].str.len()  
    df['diff_len'] = abs(df['len_q1'] - df['len_q2'])  
    df['len_word_q1'] = df['question1'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)  
    df['len_word_q2'] = df['question2'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)   
    df['common_word'] = df.apply(lambda row: common_words(row['question1'], row['question2']), axis=1)  
    #df['chr_cnt_q1'] = df['question1'].apply(cnt_chr)  
    #df['chr_cnt_q2'] = df['question2'].apply(cnt_chr)
    df['total_words'] = df['len_word_q1'] + df['len_word_q2']
    df['word_share'] = df['common_word']/df['total_words']
    return df  


def fetch_fuzzy_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    fuzzy_features = [0.0]*4
    fuzzy_features[0] = fuzz.QRatio(q1,q2)
    fuzzy_features[1] = fuzz.partial_ratio(q1,q2)
    fuzzy_features[2] = fuzz.token_sort_ratio(q1,q2)
    fuzzy_features[3] = fuzz.token_set_ratio(q1,q2)
    return fuzzy_features

def token_features(row):
    q1 = row['question1']
    q2 = row['question2']
    
    # Tokenize the questions
    q1_token = q1.split()
    q2_token = q2.split()
    
    STOP_WORDS = set(stopwords.words('english'))
    STEP = 0.0001
    new_features = [0.0] * 8  # Initialize the feature list with eight 0.0 values
    
    # If any of the questions is empty, return the initialized features
    if len(q1_token) == 0 or len(q2_token) == 0:
        return new_features
    
    # Non-stop and stop word sets for both questions
    q1_non_stop = set(word for word in q1_token if word not in STOP_WORDS)
    q2_non_stop = set(word for word in q2_token if word not in STOP_WORDS)
    
    q1_word = set(word for word in q1_token if word in STOP_WORDS)
    q2_word = set(word for word in q2_token if word in STOP_WORDS)
    
    # Calculate common tokens, non-stop words, and stop words
    common_tokens = len(set(q1_token) & set(q2_token))
    common_non_stop = len(q1_non_stop & q2_non_stop)
    common_word = len(q1_word & q2_word)
    
    # Calculate feature values
    new_features[0] = common_tokens / (min(len(q1_token), len(q2_token)) + STEP)
    new_features[1] = common_tokens / (max(len(q1_token), len(q2_token)) + STEP)
    new_features[2] = common_non_stop / (min(len(q1_non_stop), len(q2_non_stop)) + STEP)
    new_features[3] = common_non_stop / (max(len(q1_non_stop), len(q2_non_stop)) + STEP)
    new_features[4] = common_word / (min(len(q1_word), len(q2_word)) + STEP)
    new_features[5] = common_word / (max(len(q1_word), len(q2_word)) + STEP)
    
    # Check if the first and last tokens match
    new_features[6] = 1.0 if q1_token[0] == q2_token[0] else 0.0
    new_features[7] = 1.0 if q1_token[-1] == q2_token[-1] else 0.0
    
    return new_features


def distance_features(row):
    q1 = row['question1']
    q2 = row['question2']
    
    q1_token = q1.split()
    q2_token = q2.split()
    
    length_features = [0.0] * 2
    
    if len(q1_token) == 0 or len(q2_token) == 0:
        return length_features
    
    # Feature 1: Average token length
    length_features[0] = (len(q1_token) + len(q2_token)) / 2
    
    # Feature 2: Longest common substring length ratio
    strs = list(distance.lcsubstrings(q1, q2))  # Make sure `distance.lcsubstrings` is defined correctly
    if strs:  # Check if strs is not empty to avoid index errors
        length_features[1] = len(strs[0]) / (min(len(q1), len(q2)) + 0.0001)
    
    return length_features

# Load spaCy model
#nlp = spacy.load('en_core_web_sm')
exclude = string.punctuation

def preprocess(text):
    # Convert text to lowercase and strip whitespace
    text = text.lower().strip()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Replace specific symbols
    text = text.replace("%", 'percent')
    text = text.replace("$", 'dollar')
    text = text.replace("@", 'at')
    text = text.replace('[math]', " ")
    
    # Replace large numbers with suffixes
    text = re.sub(r'([0-9]+)000000000', r'\1b', text)
    text = re.sub(r'([0-9]+)000000', r'\1m', text)
    text = re.sub(r'([0-9]+)000', r'\1k', text)
    
    # Expand contractions
    contractions = { 
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he has",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how has",
        "I'd": "I had",
        "I'd've": "I would have",
        "I'll": "I shall",
        "I'll've": "I shall have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it had",
        "it'd've": "it would have",
        "it'll": "it shall",
        "it'll've": "it shall have",
        "it's": "it has",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that had",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what shall",
        "what'll've": "what shall have",
        "what're": "what are",
        "what's": "what has",
        "what've": "what have",
        "when's": "when has",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where has",
        "where've": "where have",
        "who'll": "who shall",
        "who'll've": "who shall have",
        "who's": "who has",
        "who've": "who have",
        "why's": "why has",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had",
        "you'd've": "you would have",
        "you'll": "you shall",
        "you'll've": "you shall have",
        "you're": "you are",
        "you've": "you have"
    }
    
    # Replace contractions
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', exclude))
    
    # Optional spelling correction
    text = str(TextBlob(text).correct())
    
    # Tokenize and lemmatize
    #doc = nlp(text)
    #tokens = [token.lemma_ for token in doc]
    
    return text


def question_similarity_pipeline(q1, q2):
        # Combine questions into a DataFrame for consistent processing
        data = pd.DataFrame({'question1': [q1], 'question2': [q2]})
    
        # Apply basic features
        data['question1'] = data['question1'].apply(preprocess)
        data['question2'] = data['question2'].apply(preprocess)
        data = basic_features(data)
    
        # Apply fuzzy matching features
        data[['fuzzy_qratio', 'fuzzy_partial_ratio', 'fuzzy_token_sort_ratio', 'fuzzy_token_set_ratio']] = data.apply(fetch_fuzzy_features, axis=1, result_type="expand")
    
        # Apply token-based features
        data[['common_token_min', 'common_token_max', 'common_non_stop_min', 'common_non_stop_max', 
          'common_stop_min', 'common_stop_max', 'first_token_match', 'last_token_match']] = data.apply(token_features, axis=1, result_type="expand")
    
        # Apply distance-based features
        data[['avg_token_len', 'longest_common_substring_ratio']] = data.apply(distance_features, axis=1, result_type="expand")
    
        # TF-IDF Vectorization (assuming the vectorizer is trained already)
        test_combined = [' '.join(pair.astype(str)) for pair in data[['question1', 'question2']].values]
        # Transform and convert to DataFrame
        tfv_matrix = tfv.transform(test_combined)
        tfv_df = pd.DataFrame(tfv_matrix.toarray(), columns=tfv.get_feature_names_out())
    
        # Concatenate all features
        combined_features = pd.concat([data.drop(columns=['question1', 'question2']), tfv_df], axis=1)
    
        # Model prediction
        predictions = pd.DataFrame()
        for fold_id, model in enumerate(results_xgb['oof_models']):
            predictions[fold_id] = model.predict_proba(combined_features)[:, 1]
        prediction_prob = predictions.mean(axis=1)
        prediction = "Same" if prediction_prob.iloc[0] >= 0.4 else "Different"
        return prediction, prediction_prob