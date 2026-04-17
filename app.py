import streamlit as st
import pickle
import numpy as np
import re
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


try:
    stop = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop = set(stopwords.words('english'))

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')

lemma = WordNetLemmatizer()

# ---------------------------
# Load models
# ---------------------------
w2v_model = Word2Vec.load("word2vec_model.model")

with open("spam_classifier.pkl", "rb") as f:
    model = pickle.load(f)


def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemma.lemmatize(w) for w in words if w not in stop]
    return words


def vectorize(text):
    vec = []
    for word in text:
        if word in w2v_model.wv:
            vec.append(w2v_model.wv[word])
    
    if len(vec) == 0:
        return np.zeros(w2v_model.vector_size)
    
    return np.mean(vec, axis=0)


def predict_message(msg):
    tokens = clean_text(msg)
    vec = vectorize(tokens)
    pred = model.predict([vec])[0]
    return pred, vec


st.title(" Spam Message Classifier")

user_input = st.text_area("Enter your message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        pred, vec = predict_message(user_input)

        label = " SPAM" if pred == 1 else " HAM"
        st.success(f"Prediction: {label}")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([vec])[0]
            st.write(f"HAM: {probs[0]:.4f}")
            st.write(f"SPAM: {probs[1]:.4f}")