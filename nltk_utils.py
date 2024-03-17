#######################################################
#perfect code-1
import numpy as np
import spacy
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load English tokenizer
nlp = spacy.load("en_core_web_sm")

def tokenize(sentence):
    return [token.text for token in nlp(sentence)]

def stem(word):
    return lemmatizer.lemmatize(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag


