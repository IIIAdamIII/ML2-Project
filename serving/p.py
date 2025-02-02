import pickle
with open('model/chatbot_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
