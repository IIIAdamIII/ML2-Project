from flask import Flask, render_template, request, jsonify
import pickle
import json
import random
import spacy
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob
import numpy as np
import random
import json
import warnings
warnings.filterwarnings('ignore')
app = Flask(__name__)

# Load the Random Forest model and vectorizer
with open('model/chatbot_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
# Load intents
with open('dataset/intents1.json', 'r') as f:
    intents = json.load(f)

# Function to get response from Random Forest


def rf_response(user_input):
    input_text = vectorizer.transform([user_input])
    predicted_intent = rf_model.predict(input_text)[0]
    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

# Function to get response from BERT

# Fonction de correction orthographique


def correct_text(text):
    return str(TextBlob(text).correct())


# Charger spaCy et Sentence-BERT
nlp = spacy.load("en_core_web_sm")
# Modèle pour la similarité sémantique
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
# Fonction de correction orthographique


def correct_text(text):
    return str(TextBlob(text).correct())
# Prétraitement des phrases (tokenization + lemmatisation)


def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha])


# Création des embeddings des phrases de référence
reference_texts = [intent['patterns'][0] for intent in intents['intents']]
processed_references = [preprocess(text) for text in reference_texts]
reference_embeddings = bert_model.encode(
    processed_references, convert_to_tensor=True)


def bert_prediction(user_input):
    user_input = correct_text(user_input)  # Correction orthographique
    processed_input = preprocess(user_input)
    input_embedding = bert_model.encode(
        processed_input, convert_to_tensor=True)

    # Similarité avec BERT
    similarities = util.pytorch_cos_sim(input_embedding, reference_embeddings)
    bert_match_idx = np.argmax(similarities.numpy())
    bert_response = intents['intents'][bert_match_idx]['responses'][0]

    return f"{bert_response}"

# Chatbot endpoint (Receives model choice)


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    selected_model = request.form['model']  # Get model from request

    if selected_model == "random_forest":
        response = rf_response(user_input)
    elif selected_model == "bert":
        response = bert_prediction(user_input)
    else:
        response = "Invalid model selection."

    return jsonify({"response": response})


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
