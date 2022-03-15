from flask import Flask, request, jsonify
import json
import pickle
import joblib
import string
import re
import nltk
import numpy as np

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words("german"))
stemmer = SnowballStemmer("german")

app = Flask(__name__)

# Load the trained models
model_1 = joblib.load('german_text_classifier_final') # Random Forest Classifier
model_2 = joblib.load('w2v_model_final') # word2vec
model_3 = joblib.load('label_final') #Label encoder
print(model_1)

# Text Preprocessing after each input
def re_text(text):
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('#\S+', '', text)  # remove #
    text = re.sub('@\S+', '  ', text)  # remove @
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~•▪︎➦"""), ' ', text)  # remove signs and bullets
    text = re.sub('\s+', ' ', text)  # remove extra whitespace
    text = re.sub('\n', ' ', text)
    text = re.sub('Null', '', text)
    text = "".join([i for i in text if i not in string.punctuation])
    text = ''.join([i for i in text if not i.isdigit()])
    stop_words_lambda = lambda x: ' '.join([word for word in x.split() if word not in (stop_words)])
    text = stop_words_lambda(text)
    text = word_tokenize(text)
    text = [stemmer.stem(word) for word in text]
    return text

words = set(model_2.wv.index_to_key)


def predictor(text,model):
    text = re_text(text)
    test_vect = np.array([np.array([model_2.wv[i] for i in text if i in words])] )
    x_test_vect_lst = []
    for v in test_vect:
        if v.size:
            x_test_vect_lst.append(v.mean(axis=0))
        else:
            x_test_vect_lst.append(np.zeros(100, dtype=float))
    y = model.predict(x_test_vect_lst)
    return dict((v,k) for k,v in model_3.items()).get(y[0])

@app.route('/predict')
def predict():
    # Retrieve query parameters related to this request.
    # Please enter german phrase as an api parameter
    german_phrase = request.args.get('german_phrase')
    # User should see returned label for the 'german_phrase' with status

    
    # Use the model to predict the class
    label = predictor(german_phrase, model_1)

    # # Create and send a response to the API caller
    return jsonify(status='complete', label=label)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
