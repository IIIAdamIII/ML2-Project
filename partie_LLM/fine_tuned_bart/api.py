
from flask import Flask, render_template, request
import pickle
import json
import random
from transformers import BartTokenizer, BartForConditionalGeneration
import re
app = Flask(__name__)
"""
# Load the trained model and vectorizer
with open('model/chatbot_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the intents data
with open('dataset/intents1.json', 'r') as f:
    intents = json.load(f)

def chatbot_response(user_input):
    input_text = vectorizer.transform([user_input])
    predicted_intent = best_model.predict(input_text)[0]

    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            response = random.choice(intent['responses'])
            break

    return response


"""

# Load the intents data
with open('dataset/intents1.json', 'r') as f:
    intents = json.load(f)

# Charger le modèle fine-tuné
model = BartForConditionalGeneration.from_pretrained(r"./fine_tuned_bart")
tokenizer = BartTokenizer.from_pretrained(r"./fine_tuned_bart")

def preprocess_input(input_text):
    # Supprimer les caractères spéciaux et les espaces inutiles
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', input_text)).lower()

def chatbot_response(user_input):
    user_input = preprocess_input(user_input)

    """ Génère une réponse avec BART et ajuste avec les intents JSON """
    
    # Étape 1 : Générer la réponse brute avec BART
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
    bart_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Étape 2 : Vérifier si la réponse correspond à un intent connu
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if pattern.lower() in user_input.lower():
                return random.choice(intent['responses'])

    # Étape 3 : Retourner la réponse de BART si aucun intent n'est trouvé
    return bart_response



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = chatbot_response(user_input)
    return response

if __name__ == '__main__':
    app.run(debug=True)