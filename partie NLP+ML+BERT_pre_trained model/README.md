# College Chatbot Using ML Algorithm and NLP Toolkit

The lyon 1 Chatbot is a Python-based chatbot that utilizes machine learning algorithms and natural language processing (NLP) techniques to provide automated assistance to users with UCBL-related inquiries. The chatbot aims to improve the user experience by delivering quick and accurate responses to their questions.

## Methodology

The chatbot is developed using a combination of natural language processing techniques, machine learning algorithms and pretrained models. The methodology involves data preparation, model training, and chatbot response generation. The data is preprocessed to remove noise and increase training examples using synonym replacement. Multiple classification models are trained and evaluated to find the best-performing one. The trained model is then used to predict the intent of user input, and a random response is selected from the corresponding intent's responses. The chatbot is devoloped as a web application using Flask, allowing users to interact with it in real-time.

pip install -r requirements.txt

python api.py
